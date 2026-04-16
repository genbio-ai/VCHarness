"""
Node 2 – AIDO.Cell-100M + LoRA: Multi-layer Gene-Position Fusion

Architecture:
  - AIDO.Cell-100M backbone loaded from /home/Models/AIDO.Cell-100M
  - Synthetic perturbation input: {perturbed_gene_symbol: 1.0}
    → only the queried gene has nonzero expression; all others are -1.0 (missing)
  - LoRA fine-tuning (r=16) on Q/K/V attention layers
  - Multi-layer feature fusion: weighted average of the last 6 transformer layers
  - Gene representation extracted at the perturbed gene's position in the 19264-gene output
  - Prediction head: LayerNorm + Linear(640 → 6640×3)
  - Weighted cross-entropy loss with label smoothing 0.05
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
N_LAYERS      = 18   # total transformer layers (hidden_states tuple has 19 elements)
FUSION_LAYERS = 6    # number of trailing layers to fuse

CLASS_WEIGHTS = torch.tensor([12.28, 1.12, 33.33], dtype=torch.float32)


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Per-gene macro F1 matching calc_metric.py.  pred_np: [N,3,G], labels_np: [N,G]."""
    pred_cls = pred_np.argmax(axis=1)
    f1_vals = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g];  yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class AIDOCellPerturbDataset(Dataset):
    """Stores pre-tokenised AIDO.Cell inputs + gene-position indices."""

    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        input_ids: torch.Tensor,      # [N, 19264] float32
        gene_positions: torch.Tensor,  # [N] long – index in 19264-gene space
        labels: Optional[torch.Tensor] = None,  # [N, 6640] long or None
    ):
        self.pert_ids      = pert_ids
        self.symbols       = symbols
        self.input_ids     = input_ids
        self.gene_positions = gene_positions
        self.labels        = labels

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
    pert_ids  = [b["pert_id"]       for b in batch]
    symbols   = [b["symbol"]        for b in batch]
    input_ids = torch.stack([b["input_ids"]     for b in batch])  # [B, 19264]
    gene_pos  = torch.stack([b["gene_position"] for b in batch])  # [B]
    out = {"pert_id": pert_ids, "symbol": symbols,
           "input_ids": input_ids, "gene_position": gene_pos}
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class AIDOCellDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 4,
        num_workers: int = 2,
    ):
        super().__init__()
        self.data_dir        = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers     = num_workers

    def setup(self, stage: Optional[str] = None):
        # ── Load tokenizer (DDP-safe barrier) ─────────────────────────────
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)

        # ── Helper: tokenize a list of gene symbols ────────────────────────
        def tokenize_symbols(symbols: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
            """Returns (input_ids [N, 19264], gene_positions [N])."""
            batch_input = [{"gene_names": [sym], "expression": [1.0]} for sym in symbols]
            tok_out = tokenizer(batch_input, return_tensors="pt")
            ids = tok_out["input_ids"]  # [N, 19264] float32
            # Gene position: the unique location where expression > 0
            # (all others are -1.0).  argmax over >0 mask gives the index.
            gene_pos = (ids > 0.5).float().argmax(dim=1).long()  # [N]
            return ids, gene_pos

        # ── Load splits ────────────────────────────────────────────────────
        def load_split(fname: str, has_label: bool):
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            symbols  = df["symbol"].tolist()
            pert_ids = df["pert_id"].tolist()
            ids, gpos = tokenize_symbols(symbols)
            labels = None
            if has_label and "label" in df.columns:
                rows = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return AIDOCellPerturbDataset(pert_ids, symbols, ids, gpos, labels)

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds, batch_size=self.micro_batch_size,
            shuffle=shuffle, collate_fn=collate_fn,
            num_workers=self.num_workers, pin_memory=True,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Model ────────────────────────────────────────────────────────────────────

class AIDOCellPerturbModel(nn.Module):
    """AIDO.Cell-100M + LoRA backbone with multi-layer gene-position fusion head."""

    def __init__(self, n_genes_out: int = N_GENES_OUT, n_classes: int = N_CLASSES):
        super().__init__()
        # Load backbone
        backbone = AutoModel.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False

        # AIDO.Cell does not implement get_input_embeddings(), which PEFT calls
        # via enable_input_require_grads() unconditionally in PeftModel.__init__.
        # Patch enable_input_require_grads to register a hook on the gene embedding
        # layer instead, ensuring gradient flow through the frozen backbone.
        def _safe_enable_input_require_grads():
            def _make_inputs_require_grad(module, input, output):
                if isinstance(output, torch.Tensor):
                    output.requires_grad_(True)
            backbone.bert.gene_embedding.register_forward_hook(_make_inputs_require_grad)
        backbone.enable_input_require_grads = _safe_enable_input_require_grads

        # Apply LoRA to Q/K/V in ALL 18 layers
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["query", "key", "value"],
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Enable gradient checkpointing after PEFT wrapping for memory efficiency.
        # For LoRA, peak memory is ~3.41 GiB either way, but this is best practice.
        self.backbone.base_model.model.config.use_cache = False
        self.backbone.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA params to float32
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Learnable layer-fusion weights for last FUSION_LAYERS layers
        self.layer_weights = nn.Parameter(torch.zeros(FUSION_LAYERS))

        # Prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Linear(HIDDEN_SIZE, n_genes_out * n_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,      # [B, 19264] float32
        gene_positions: torch.Tensor,  # [B] long
    ) -> torch.Tensor:
        """Returns logits [B, 3, 6640]."""
        # AIDO.Cell expects attention_mask (all ones is fine – model overrides anyway)
        attn_mask = torch.ones(input_ids.shape[0], input_ids.shape[1],
                               dtype=torch.long, device=input_ids.device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )

        # hidden_states: tuple of 19 tensors [B, 19266, 640]
        # indices 0..18; we use the last FUSION_LAYERS = [13..18]
        hidden_states = torch.stack(
            [out.hidden_states[i].float() for i in range(N_LAYERS - FUSION_LAYERS + 1, N_LAYERS + 1)],
            dim=0,
        )  # [FUSION_LAYERS, B, 19266, 640]

        # Weighted combination across fusion layers
        weights = torch.softmax(self.layer_weights, dim=0)  # [FUSION_LAYERS]
        fused = (hidden_states * weights[:, None, None, None]).sum(0)  # [B, 19266, 640]

        # Extract the representation at each sample's perturbed gene position
        B = fused.shape[0]
        gene_repr = fused[torch.arange(B, device=fused.device), gene_positions, :]  # [B, 640]

        # Decode to DEG signature
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
    dist.all_gather(gp, p);  dist.all_gather(gl, l)

    rp = torch.cat([gp[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    rl = torch.cat([gl[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    return rp, rl


# ─── LightningModule ──────────────────────────────────────────────────────────

class AIDOCellLitModule(pl.LightningModule):

    def __init__(
        self,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
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
        self.model = AIDOCellPerturbModel()
        self.register_buffer("class_weights", CLASS_WEIGHTS)

    def forward(self, input_ids, gene_positions):
        return self.model(input_ids, gene_positions)

    def _loss(self, logits, labels):
        # Reshape to 2D before cross_entropy to use the deterministic nll_loss kernel
        # (avoids nll_loss2d which has no deterministic CUDA implementation).
        # logits: [B, 3, 6640] -> [B*6640, 3];  labels: [B, 6640] -> [B*6640]
        logits_2d = logits.float().permute(0, 2, 1).reshape(-1, N_CLASSES)
        labels_1d = labels.reshape(-1)
        return F.cross_entropy(
            logits_2d, labels_1d,
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
        # sync_dist=True: all ranks hold the same global f1 (already all_gathered above),
        # so averaging across ranks simply returns the same value. This silences Lightning's
        # epoch-level distributed logging warning.
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
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate by pert_id using index tracking.
            # DDP DistributedSampler may pad the dataset with duplicate samples
            # to ensure even distribution across GPUs; the metric script requires
            # unique pert_ids in the prediction file.
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
            self.print(f"[Node2] Saved {len(dedup_indices)} test predictions → {pred_path}")
            if all_labels.any():
                dedup_probs  = all_probs_np[dedup_indices]
                dedup_labels = all_labels[dedup_indices].numpy()
                f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
                self.print(f"[Node2] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
        )
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
    p = argparse.ArgumentParser(description="Node 2 – AIDO.Cell-100M LoRA")
    p.add_argument("--data-dir",            type=str,   default="data")
    p.add_argument("--lr",                  type=float, default=2e-4)
    p.add_argument("--weight-decay",        type=float, default=1e-4)
    p.add_argument("--label-smoothing",     type=float, default=0.05)
    p.add_argument("--micro-batch-size",    type=int,   default=4)
    p.add_argument("--global-batch-size",   type=int,   default=32)
    p.add_argument("--max-epochs",          type=int,   default=150)
    p.add_argument("--patience",            type=int,   default=15)
    p.add_argument("--num-workers",         type=int,   default=2)
    p.add_argument("--val-check-interval",  type=float, default=1.0,
                   help="Validation check interval (fraction of epoch or number of steps)")
    p.add_argument("--debug-max-step",      type=int,   default=None,
                   help="Limit training/val/test to this many steps (debug only, default: full run)")
    p.add_argument("--fast-dev-run",        action="store_true", default=False,
                   help="Run 1 batch through each stage for unit testing")
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    dm  = AIDOCellDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = AIDOCellLitModule(args.lr, args.weight_decay, args.label_smoothing)

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
        max_steps   = args.debug_max_step
        limit_train = args.debug_max_step
        limit_val   = 2;  limit_test = 2
    if args.fast_dev_run:
        fast_dev_run = True

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    # Always use DDPStrategy for consistent multi-GPU behaviour; works with n_gpus=1 too
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
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)

    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 2 – AIDO.Cell-100M LoRA\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
