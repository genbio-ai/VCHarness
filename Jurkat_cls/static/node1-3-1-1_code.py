#!/usr/bin/env python3
"""
Node 1-3-1-1: Frozen AIDO.Cell-10M + Pre-computed STRING PPI + Char CNN + Smaller 3-stage MLP Head
====================================================================================================
Improvements over parent node1-3-1 (test F1=0.4420):

1. Smaller head bottleneck: 896→384→128→19920 (vs 896→384→256→19920)
   - Final linear layer: 128×19920=2.55M params (vs 256×19920=5.1M)
   - Total head: ~3.2M (vs ~5.5M), reduces overfitting

2. Milder class weights: [1.5, 1.0, 2.0] (vs [3.0, 1.0, 5.0])
   - Reduces val_f1 inflation from heavy class re-weighting
   - Smaller val-test gap (inspired by node3-2 with gap=0.025)

3. Lower focal gamma: 1.5 (vs 2.0) — less aggressive minority-class focus

4. Lower label smoothing: 0.05 (vs 0.10) — less regularization

5. Dropout: 0.4 (vs 0.5) — appropriate for smaller head

6. Save top-5 checkpoints (vs top-3) — more candidates for averaging

7. Checkpoint averaging: Average top-3 checkpoints by weight at test time

8. Early stopping patience: 40 (vs 30) — allow more training time
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

N_GENES = 6_640
N_CLASSES = 3
AIDO_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")

# Milder class weights to reduce val_f1 inflation and val-test gap
CLASS_WEIGHTS = torch.tensor([1.5, 1.0, 2.0], dtype=torch.float32)

CHAR_VOCAB = {c: i + 1 for i, c in enumerate(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.abcdefghijklmnopqrstuvwxyz"
)}
CHAR_VOCAB["<pad>"] = 0
MAX_SYM_LEN = 16

AIDO_HIDDEN_DIM = 256
STRING_HIDDEN_DIM = 256
CHAR_CNN_DIM = 128
FUSION_DIM = AIDO_HIDDEN_DIM * 2 + STRING_HIDDEN_DIM + CHAR_CNN_DIM  # 896


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, targets, weight=w,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return ((1.0 - pt) ** self.gamma * ce).mean()


def encode_symbol(symbol: str, max_len: int = MAX_SYM_LEN) -> torch.Tensor:
    sym = symbol.upper()[:max_len]
    ids = [CHAR_VOCAB.get(c, 0) for c in sym]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    y_hat = y_pred.argmax(axis=1)
    f1_vals: List[float] = []
    for g in range(y_true_remapped.shape[1]):
        yt = y_true_remapped[:, g]
        yp = y_hat[:, g]
        present = np.array([(yt == c).any() for c in range(3)])
        pf1 = sk_f1_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


def precompute_features(data_dir: Path, cache_dir: Path,
                        precompute_batch_size: int = 8) -> Dict[str, np.ndarray]:
    cache_path = cache_dir / "feature_cache.npz"
    if cache_path.exists():
        print(f"[PreCompute] Loading cached features from {cache_path}")
        return dict(np.load(str(cache_path), allow_pickle=True))

    print("[PreCompute] Computing AIDO + STRING features (this runs once)...")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

    gene_id_to_pos: Dict[str, int] = {}
    if hasattr(tokenizer, "gene_id_to_index"):
        gene_id_to_pos = {k: int(v) for k, v in tokenizer.gene_id_to_index.items()}
    elif hasattr(tokenizer, "gene_to_index"):
        gene_id_to_pos = {k: int(v) for k, v in tokenizer.gene_to_index.items()}

    aido_model = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
    aido_model = aido_model.to(torch.bfloat16).eval().to(device)

    string_model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
    string_model = string_model.eval().to(device)

    graph = torch.load(str(STRING_GNN_DIR / "graph_data.pt"), map_location=device)
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    string_name_to_idx = {n: i for i, n in enumerate(node_names)}

    edge_index = graph["edge_index"]
    edge_weight = graph.get("edge_weight")
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    with torch.no_grad():
        string_outputs = string_model(edge_index=edge_index, edge_weight=edge_weight)
    all_string_embs = string_outputs.last_hidden_state.float().cpu().numpy()

    results: Dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        tsv_path = data_dir / f"{split}.tsv"
        if not tsv_path.exists():
            continue
        df = pd.read_csv(tsv_path, sep="\t")
        pert_ids: List[str] = df["pert_id"].tolist()
        N = len(pert_ids)
        print(f"[PreCompute] Processing '{split}' split: {N} samples")

        aido_feats_list: List[np.ndarray] = []
        for start in range(0, N, precompute_batch_size):
            batch_ids = pert_ids[start:start + precompute_batch_size]
            batch_data = [
                {"gene_ids": [pid.split(".")[0]], "expression": [1.0]}
                for pid in batch_ids
            ]
            batch_inputs = tokenizer(batch_data, return_tensors="pt")
            input_ids = batch_inputs["input_ids"].to(device)
            attention_mask = batch_inputs["attention_mask"].to(device)

            with torch.no_grad():
                outputs = aido_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state.float()
            gene_hidden = last_hidden[:, :19264, :]

            mean_emb = gene_hidden.mean(dim=1)

            gene_pos_embs: List[torch.Tensor] = []
            for j, pid in enumerate(batch_ids):
                eid = pid.split(".")[0]
                pos = gene_id_to_pos.get(eid)
                if pos is not None:
                    pos_clamped = max(0, min(int(pos), 19263))
                    gene_pos_embs.append(gene_hidden[j, pos_clamped, :])
                else:
                    gene_pos_embs.append(mean_emb[j])

            gene_pos_emb = torch.stack(gene_pos_embs)
            dual_pool = torch.cat([gene_pos_emb, mean_emb], dim=-1)
            aido_feats_list.append(dual_pool.cpu().float().numpy())

            if (start // precompute_batch_size) % 20 == 0:
                print(f"    AIDO: {start + len(batch_ids)}/{N} samples done")

        aido_feats = np.concatenate(aido_feats_list, axis=0)

        string_feats = np.zeros((N, STRING_HIDDEN_DIM), dtype=np.float32)
        string_found = 0
        for j, pid in enumerate(pert_ids):
            eid = pid.split(".")[0]
            idx = string_name_to_idx.get(eid)
            if idx is not None:
                string_feats[j] = all_string_embs[idx]
                string_found += 1
        print(f"    STRING coverage: {string_found}/{N} ({100*string_found/N:.1f}%) genes found")

        results[f"{split}_aido"] = aido_feats.astype(np.float32)
        results[f"{split}_string"] = string_feats.astype(np.float32)

    np.savez(str(cache_path), **results)
    print(f"[PreCompute] Feature cache saved → {cache_path}")

    del aido_model, string_model
    torch.cuda.empty_cache()

    return results


class PrecomputedDEGDataset(Dataset):
    def __init__(self, df, aido_feats, string_feats, is_test=False):
        self.pert_ids = df["pert_id"].tolist()
        self.symbols = df["symbol"].tolist()
        self.aido_feats = torch.from_numpy(aido_feats.astype(np.float32))
        self.string_feats = torch.from_numpy(string_feats.astype(np.float32))
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            self.labels = torch.tensor(
                np.array(raw_labels, dtype=np.int8) + 1, dtype=torch.long
            )
        else:
            self.labels = None

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "aido_feats": self.aido_feats[idx],
            "string_feats": self.string_feats[idx],
            "sym_ids": encode_symbol(self.symbols[idx]),
        }
        if not self.is_test:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    result = {}
    for key in batch[0]:
        if key == "pert_id":
            result[key] = [item[key] for item in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], int):
            result[key] = torch.tensor([item[key] for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    return result


class DEGDataModule(LightningDataModule):
    def __init__(self, data_dir, cache_dir, micro_batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.test_pert_ids = []
        self.test_symbols = []
        self._cache = None

    def setup(self, stage=None):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        cache_path = self.cache_dir / "feature_cache.npz"

        if local_rank == 0 and not cache_path.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            precompute_features(self.data_dir, self.cache_dir)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self._cache is None:
            self._cache = dict(np.load(str(cache_path), allow_pickle=True))

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PrecomputedDEGDataset(
                train_df, self._cache["train_aido"], self._cache["train_string"]
            )
            self.val_ds = PrecomputedDEGDataset(
                val_df, self._cache["val_aido"], self._cache["val_string"]
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PrecomputedDEGDataset(
                test_df, self._cache["test_aido"], self._cache["test_string"], is_test=True
            )
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.micro_batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.micro_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn)


class SymbolCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, out_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embed_dim, out_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, out_dim, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(out_dim * 2)
        self.out_dim = out_dim * 2

    def forward(self, ids):
        x = self.embedding(ids).transpose(1, 2)
        f3 = F.gelu(self.conv3(x)).max(dim=-1).values
        f5 = F.gelu(self.conv5(x)).max(dim=-1).values
        return self.norm(torch.cat([f3, f5], dim=-1))


class DEGModel(nn.Module):
    def __init__(self, hidden_dim=384, dropout=0.4):
        super().__init__()
        self.symbol_cnn = SymbolCNN(vocab_size=len(CHAR_VOCAB), embed_dim=32, out_dim=64)
        self.string_missing_emb = nn.Parameter(torch.zeros(STRING_HIDDEN_DIM))
        self.aido_norm = nn.LayerNorm(AIDO_HIDDEN_DIM * 2)
        self.string_norm = nn.LayerNorm(STRING_HIDDEN_DIM)

        # 3-stage MLP head: 896 → hidden_dim → 128 → N_CLASSES*N_GENES
        # Smaller bottleneck (128 vs 256) reduces final linear from 5.1M to 2.55M params
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Linear(FUSION_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, N_CLASSES * N_GENES),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.string_missing_emb, mean=0.0, std=0.01)

    def forward(self, aido_feats, string_feats, sym_ids):
        string_is_zero = (string_feats.abs().sum(dim=-1, keepdim=True) == 0)
        missing_emb = self.string_missing_emb.unsqueeze(0).expand(string_feats.shape[0], -1)
        string_feats = torch.where(string_is_zero, missing_emb.to(string_feats.dtype), string_feats)
        aido_feats = self.aido_norm(aido_feats.float())
        string_feats = self.string_norm(string_feats.float())
        sym_feats = self.symbol_cnn(sym_ids)
        fused = torch.cat([aido_feats, string_feats, sym_feats], dim=-1)
        logits = self.head(fused)
        return logits.view(-1, N_CLASSES, N_GENES)


class DEGLightningModule(LightningModule):
    def __init__(self, hidden_dim=384, dropout=0.4, lr=1e-3, weight_decay=5e-2,
                 gamma_focal=1.5, label_smoothing=0.05, t_max=150, eta_min=1e-6):
        super().__init__()
        self.save_hyperparameters()
        self.model = None
        self.criterion = None
        self._val_preds = []
        self._val_labels = []
        self._val_indices = []
        self._test_preds = []
        self._test_indices = []
        self._test_pert_ids = []
        self._test_symbols = []
        self._n_genes = N_GENES

    def setup(self, stage=None):
        if self.model is None:
            self.model = DEGModel(hidden_dim=self.hparams.hidden_dim, dropout=self.hparams.dropout)
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = param.data.float()

        if stage == "test":
            if self.trainer is not None and hasattr(self.trainer, "datamodule"):
                dm = self.trainer.datamodule
                if dm is not None and hasattr(dm, "test_pert_ids"):
                    self._test_pert_ids = dm.test_pert_ids
                    self._test_symbols = dm.test_symbols

    def _forward(self, batch):
        return self.model(aido_feats=batch["aido_feats"], string_feats=batch["string_feats"],
                          sym_ids=batch["sym_ids"])

    def _compute_loss(self, logits, labels):
        B, C, G = logits.shape
        return self.criterion(logits.permute(0, 2, 1).reshape(-1, C), labels.reshape(-1))

    def training_step(self, batch, batch_idx):
        logits = self._forward(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self._forward(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return

        # Compute local F1 on each GPU (lightweight scalar, no heavy gathering)
        local_preds = torch.cat(self._val_preds, dim=0).numpy()
        local_labels = torch.cat(self._val_labels, dim=0).numpy()
        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        local_f1 = compute_deg_f1(local_preds, local_labels)
        f1_tensor = torch.tensor(local_f1, dtype=torch.float32, device=self.device)

        world_size = self.trainer.world_size if self.trainer.world_size else 1
        if world_size > 1:
            import torch.distributed as dist
            dist.all_reduce(f1_tensor, op=dist.ReduceOp.SUM)
            f1_tensor = f1_tensor / world_size

        self.log("val_f1", f1_tensor.item(), prog_bar=True, sync_dist=False)

    def test_step(self, batch, batch_idx):
        logits = self._forward(batch)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())

    def on_test_epoch_end(self):
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, dim=0)
        local_idx = torch.cat(self._test_indices, dim=0)

        all_preds = self.all_gather(local_preds)
        all_idx = self.all_gather(local_idx)

        self._test_preds.clear()
        self._test_indices.clear()

        if self.trainer.is_global_zero:
            preds = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
            idxs = all_idx.view(-1).cpu().numpy()

            unique_pos = np.unique(idxs, return_index=True)[1]
            preds = preds[unique_pos]
            sorted_idxs = idxs[unique_pos]

            order = np.argsort(sorted_idxs)
            preds = preds[order]
            final_idxs = sorted_idxs[order]

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "test_predictions.tsv"

            rows = []
            for rank_i, orig_i in enumerate(final_idxs):
                rows.append({
                    "idx": self._test_pert_ids[int(orig_i)],
                    "input": self._test_symbols[int(orig_i)],
                    "prediction": json.dumps(preds[rank_i].tolist()),
                })
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {out_path} ({len(rows)} rows)")

    def configure_optimizers(self):
        head_params = list(self.model.head.parameters())
        other_params = [
            p for p in self.model.parameters()
            if p.requires_grad and not any(p is hp for hp in head_params)
        ]
        param_groups = [
            {"params": head_params, "lr": self.hparams.lr, "weight_decay": self.hparams.weight_decay},
            {"params": other_params, "lr": self.hparams.lr * 0.5, "weight_decay": self.hparams.weight_decay},
        ]
        opt = torch.optim.AdamW(param_groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.t_max, eta_min=self.hparams.eta_min,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_state_dict:
                    trainable_state_dict[key] = full_state_dict[key]
        for name, buffer in self.named_buffers():
            key = prefix + name
            if key in full_state_dict:
                trainable_state_dict[key] = full_state_dict[key]
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(f"Saving checkpoint: {trainable_params}/{total_params} params ({100*trainable_params/total_params:.2f}%), plus {total_buffers} buffer values")
        return trainable_state_dict

    def load_state_dict(self, state_dict, strict=True):
        full_state_keys = set(super().state_dict().keys())
        trainable_keys = {name for name, param in self.named_parameters() if param.requires_grad}
        buffer_keys = {name for name, _ in self.named_buffers() if name in full_state_keys}
        expected_keys = trainable_keys | buffer_keys
        missing_keys = [k for k in expected_keys if k not in state_dict]
        unexpected_keys = [k for k in state_dict if k not in expected_keys]
        if missing_keys:
            self.print(f"Warning: Missing checkpoint keys: {missing_keys[:5]}...")
        if unexpected_keys:
            self.print(f"Warning: Unexpected checkpoint keys: {unexpected_keys[:5]}...")
        loaded_trainable = len([k for k in state_dict if k in trainable_keys])
        loaded_buffers = len([k for k in state_dict if k in buffer_keys])
        self.print(f"Loading checkpoint: {loaded_trainable} trainable params and {loaded_buffers} buffers")
        return super().load_state_dict(state_dict, strict=False)


def average_checkpoints(ckpt_paths: List[str], model_module: DEGLightningModule) -> DEGLightningModule:
    """Average parameters from multiple checkpoints for improved generalization."""
    if not ckpt_paths:
        return model_module
    avg_state = None
    count = 0
    for path in ckpt_paths:
        if not Path(path).exists():
            print(f"[CkptAvg] Checkpoint not found, skipping: {path}")
            continue
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        if avg_state is None:
            avg_state = {k: v.float().clone() for k, v in state.items()}
        else:
            for k in avg_state:
                if k in state:
                    avg_state[k] += state[k].float()
        count += 1
        print(f"[CkptAvg] Loaded checkpoint {count}: {path}")
    if avg_state is None or count == 0:
        print("[CkptAvg] No valid checkpoints found, using current model state.")
        return model_module
    for k in avg_state:
        avg_state[k] /= count
    print(f"[CkptAvg] Averaged {count} checkpoints.")
    model_module.load_state_dict(avg_state)
    return model_module


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--gamma-focal", type=float, default=1.5)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--t-max", type=int, default=150)
    p.add_argument("--eta-min", type=float, default=1e-6)
    p.add_argument("--early-stopping-patience", type=int, default=40)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    return p.parse_args()


def main():
    pl.seed_everything(0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit_train = limit_val = limit_test = 1.0
    if args.debug_max_step is not None:
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
        limit_test = args.debug_max_step

    val_check_interval = args.val_check_interval if (
        args.debug_max_step is None and not args.fast_dev_run
    ) else 1.0

    if n_gpus == 1:
        strategy = SingleDeviceStrategy(device="cuda:0")
    else:
        strategy = DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=300))

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-3-1-1-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=5,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(monitor="val_f1", mode="max", patience=args.early_stopping_patience, verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    datamodule = DEGDataModule(data_dir=args.data_dir, cache_dir=str(cache_dir),
                               micro_batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model_module = DEGLightningModule(
        hidden_dim=args.hidden_dim, dropout=args.dropout, lr=args.lr,
        weight_decay=args.weight_decay, gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing, t_max=args.t_max, eta_min=args.eta_min,
    )

    trainer.fit(model_module, datamodule=datamodule)

    # Test phase: use checkpoint averaging for non-debug runs
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        # Average top checkpoints before testing (zero-risk improvement)
        ckpt_paths = sorted(checkpoint_cb.best_k_models.keys(),
                            key=lambda p: checkpoint_cb.best_k_models[p],
                            reverse=True)  # sort by val_f1 descending
        top_k = min(3, len(ckpt_paths))
        if top_k > 0:
            print(f"[CkptAvg] Averaging top-{top_k} checkpoints: {ckpt_paths[:top_k]}")
            # Load best checkpoint first to initialize model state, then average all top-k
            best_ckpt = torch.load(checkpoint_cb.best_model_path, map_location='cpu')
            best_state = best_ckpt.get('state_dict', best_ckpt)
            model_module.load_state_dict(best_state)
            # average_checkpoints includes all paths (including best) → clean averaging
            model_module = average_checkpoints(ckpt_paths[:top_k], model_module)
            test_results = trainer.test(model_module, datamodule=datamodule)
        else:
            test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        with open(score_path, "w") as f:
            f.write(f"Test results: {json.dumps(test_results, indent=2)}\n")
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
