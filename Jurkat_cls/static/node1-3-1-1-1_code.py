#!/usr/bin/env python3
"""
Node 1-2: Hybrid LoRA + STRING PPI + Char CNN - Escaping the Frozen Backbone Ceiling
======================================================================================
Key improvement from parent node1-3-1-1 (test F1=0.4187):

The parent suffered from a hard representational ceiling due to using a frozen AIDO.Cell-10M
backbone. Feedback analysis confirmed the frozen-backbone paradigm has saturated at ~0.44 F1
across 8+ generations. This node pivots to LoRA fine-tuning, specifically:

1. AIDO.Cell-10M with LoRA r=4 (all 8 transformer layers, QKV projections)
   - Enables backbone adaptation with minimal overfitting risk (low rank)
   - All 8 layers (not just last 4) for broader representation adaptation
   - Total LoRA params: ~18K (4 params/sample, extremely conservative)

2. Aggressive class weights [5.0, 1.0, 10.0] + focal gamma=2.0
   - Restores strong minority-class emphasis (tree-best node3-2 used these exact values)
   - Parent's milder weights [1.5, 1.0, 2.0] suppressed minority class learning

3. ReduceLROnPlateau monitoring val_loss (not val_f1)
   - val_f1 oscillates ±0.005 due to focal loss re-weighting (fails to trigger scheduler)
   - val_loss is a more stable signal for LR reduction decisions
   - Proven by node3-2 (tree-best) and node2-3-1-1 (second-best at 0.4555)

4. Single best checkpoint (NO averaging)
   - Averaging was confirmed harmful in parent: degraded test F1 by averaging diverse
     optimization states at different LR phases in CosineAnnealing

5. Restored 384-dim MLP head bottleneck
   - Parent reduced to 128-dim causing severe representational bottleneck
   - 384-dim head matches tree-best node3-2's proven configuration

6. Architecture: 4-source feature fusion
   - AIDO.Cell-10M dual-pool (LoRA, 512-dim): gene_pos_emb(256) + mean_pool(256)
   - STRING GNN PPI embeddings (frozen, 256-dim): pre-computed topology features
   - Character-level CNN on gene symbols (128-dim): multi-scale 3+5 kernels
   - Total fusion: 896-dim → 384 → 19920
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
from peft import LoraConfig, get_peft_model, TaskType

N_GENES = 6_640
N_CLASSES = 3
AIDO_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")

# Aggressive class weights matching tree-best node3-2 [5.0, 1.0, 10.0]
# Class mapping: 0=down-regulated(-1), 1=unchanged(0), 2=up-regulated(+1)
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)

CHAR_VOCAB = {c: i + 1 for i, c in enumerate(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.abcdefghijklmnopqrstuvwxyz"
)}
CHAR_VOCAB["<pad>"] = 0
MAX_SYM_LEN = 16

AIDO_HIDDEN_DIM = 256   # AIDO.Cell-10M hidden size
STRING_HIDDEN_DIM = 256
CHAR_CNN_DIM = 128      # 2 kernels × 64 output each
# Total fusion dimension: 512 (AIDO dual-pool) + 256 (STRING) + 128 (Char CNN) = 896
FUSION_DIM = AIDO_HIDDEN_DIM * 2 + STRING_HIDDEN_DIM + CHAR_CNN_DIM  # 896


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.05):
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


def precompute_features(data_dir: Path, cache_dir: Path) -> Dict[str, np.ndarray]:
    """
    Pre-compute frozen STRING GNN PPI features for all splits.

    Unlike the frozen-backbone version, we do NOT pre-compute AIDO features here.
    AIDO features are computed live during training via the LoRA backbone to enable
    gradient flow. Only STRING GNN features (which remain fully frozen) are cached.
    """
    cache_path = cache_dir / "feature_cache.npz"
    if cache_path.exists():
        print(f"[PreCompute] Loading cached features from {cache_path}")
        return dict(np.load(str(cache_path), allow_pickle=True))

    print("[PreCompute] Computing STRING GNN features (this runs once)...")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

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

    # Dummy placeholder for AIDO features (shape doesn't matter — not used in training)
    # We store zeros to satisfy backward compatibility with cache loading
    results: Dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        tsv_path = data_dir / f"{split}.tsv"
        if not tsv_path.exists():
            continue
        df = pd.read_csv(tsv_path, sep="\t")
        pert_ids: List[str] = df["pert_id"].tolist()
        N = len(pert_ids)
        print(f"[PreCompute] Processing '{split}' split: {N} samples")

        string_feats = np.zeros((N, STRING_HIDDEN_DIM), dtype=np.float32)
        string_found = 0
        for j, pid in enumerate(pert_ids):
            eid = pid.split(".")[0]
            idx = string_name_to_idx.get(eid)
            if idx is not None:
                string_feats[j] = all_string_embs[idx]
                string_found += 1
        print(f"    STRING coverage: {string_found}/{N} ({100*string_found/N:.1f}%) genes found")

        results[f"{split}_string"] = string_feats.astype(np.float32)

    np.savez(str(cache_path), **results)
    print(f"[PreCompute] STRING GNN feature cache saved → {cache_path}")

    del string_model
    torch.cuda.empty_cache()

    return results


class DEGDataset(Dataset):
    """
    Dataset for DEG prediction with LoRA fine-tuning.

    Unlike the frozen-backbone version, we do NOT store AIDO features here.
    Instead, pert_id strings are stored so that live tokenization can happen
    during training (required for LoRA gradient flow through the backbone).

    Only STRING GNN PPI features are pre-computed and stored (they remain frozen).
    """
    def __init__(self, df, string_feats, is_test=False):
        self.pert_ids = df["pert_id"].tolist()
        self.symbols = df["symbol"].tolist()
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
    def __init__(self, data_dir, cache_dir, micro_batch_size=8, num_workers=4):
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

        # Pre-compute STRING GNN features (frozen) if not cached
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
            # DEGDataset stores pert_ids for live tokenization + STRING feats (frozen)
            self.train_ds = DEGDataset(train_df, self._cache["train_string"])
            self.val_ds = DEGDataset(val_df, self._cache["val_string"])

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = DEGDataset(test_df, self._cache["test_string"], is_test=True)
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
    """Multi-scale character-level CNN for gene symbol encoding."""
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


class LoRAAdapter(nn.Module):
    """
    AIDO.Cell-10M with LoRA fine-tuning (r=4, all 8 layers, QKV projections).

    Key design choices:
    - r=4 (very low rank) → ~18K trainable params, ~12 params/sample, minimal overfitting
    - All 8 layers (not just last N) → broader adaptation capacity per rank unit
    - alpha=8 (2× r) → standard scaling that prevents magnitude amplification
    - lora_dropout=0.1 → light regularization on adaptation paths

    During training, the backbone processes each input sample and extracts:
    1. Per-gene positional embedding at the perturbed gene's position
    2. Global mean-pool over all 19264 gene positions
    Concatenated: 256+256 = 512-dim AIDO feature vector
    """
    def __init__(self, model_dir: str, lora_r: int = 4, lora_alpha: int = 8,
                 lora_dropout: float = 0.1):
        super().__init__()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        # Build gene_id_to_pos lookup
        self.gene_id_to_pos: Dict[str, int] = {}
        if hasattr(self.tokenizer, "gene_id_to_index"):
            self.gene_id_to_pos = {k: int(v) for k, v in self.tokenizer.gene_id_to_index.items()}
        elif hasattr(self.tokenizer, "gene_to_index"):
            self.gene_id_to_pos = {k: int(v) for k, v in self.tokenizer.gene_to_index.items()}

        base_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)

        # Disable FlashAttention when using LoRA — PEFT LoRA targets attention.self.{query,key,value}
        # but AIDO.Cell's attention uses flash_self for FlashAttention, bypassing LoRA modifications.
        # With flash_self bypassed, only frozen backbone runs and loss requires_grad=False.
        # Disabling flash_self forces the model through LoRA-modified self.self path.
        if hasattr(base_model, "config"):
            base_model.config._use_flash_attention_2 = False

        # Apply LoRA to QKV projections in all 8 transformer layers
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,  # all layers
            bias="none",
        )
        self.backbone = get_peft_model(base_model, lora_cfg)
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast model to bf16 for memory efficiency, preserving requires_grad
        # NOTE: calling .to(dtype) without preserving requires_grad destroys it!
        for name, param in self.backbone.named_parameters():
            orig_requires_grad = param.requires_grad
            param.data = param.data.to(dtype=torch.bfloat16)
            param.requires_grad = orig_requires_grad

        # Cast LoRA params to float32 for stable optimization (preserving requires_grad)
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                orig_requires_grad = param.requires_grad
                param.data = param.data.float()
                param.requires_grad = orig_requires_grad

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"[LoRAAdapter] Trainable: {trainable:,}/{total:,} params "
              f"({100*trainable/total:.3f}%)")

    def forward(self, input_ids, attention_mask, pert_ids: List[str]):
        """
        Args:
            input_ids: [B, 19264] float32 expression values
            attention_mask: [B, 19264] int64
            pert_ids: list of B Ensembl gene ID strings

        Returns:
            dual_pool: [B, 512] concatenated (gene_pos_emb, mean_emb)
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, 19266, 256]
        gene_hidden = last_hidden[:, :19264, :].float()  # [B, 19264, 256]

        mean_emb = gene_hidden.mean(dim=1)  # [B, 256]

        # Extract positional embedding at the perturbed gene's position
        gene_pos_embs: List[torch.Tensor] = []
        for j, pid in enumerate(pert_ids):
            eid = pid.split(".")[0]
            pos = self.gene_id_to_pos.get(eid)
            if pos is not None:
                pos_clamped = max(0, min(int(pos), 19263))
                gene_pos_embs.append(gene_hidden[j, pos_clamped, :])
            else:
                gene_pos_embs.append(mean_emb[j])

        gene_pos_emb = torch.stack(gene_pos_embs)  # [B, 256]
        dual_pool = torch.cat([gene_pos_emb, mean_emb], dim=-1)  # [B, 512]
        return dual_pool


class DEGModel(nn.Module):
    """
    4-source feature fusion model for DEG prediction.

    Architecture:
    - AIDO.Cell-10M with LoRA (512-dim): dual pool = gene_pos_emb(256) + mean(256)
    - STRING GNN PPI (256-dim): frozen pre-computed topology embeddings
    - Character-level CNN (128-dim): multi-scale gene symbol encoder
    - Fusion (896-dim) → LayerNorm → Linear(896→384) → GELU → Dropout → Linear(384→19920)
    """
    def __init__(self, lora_r: int = 4, lora_alpha: int = 8, lora_dropout: float = 0.1,
                 hidden_dim: int = 384, dropout: float = 0.4):
        super().__init__()
        self.lora_adapter = LoRAAdapter(
            model_dir=AIDO_MODEL_DIR,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.symbol_cnn = SymbolCNN(vocab_size=len(CHAR_VOCAB), embed_dim=32, out_dim=64)
        self.string_missing_emb = nn.Parameter(torch.zeros(STRING_HIDDEN_DIM))
        self.string_norm = nn.LayerNorm(STRING_HIDDEN_DIM)

        # Single-stage MLP head matching tree-best node3-2's proven configuration
        # 896 → 384 → 19920 (N_CLASSES * N_GENES = 3 * 6640)
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Linear(FUSION_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, N_CLASSES * N_GENES),
        )
        self._init_weights()

    def _init_weights(self):
        # Only init head and symbol_cnn weights — do NOT touch LoRA backbone weights
        for m in [self.head, self.symbol_cnn]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.normal_(self.string_missing_emb, mean=0.0, std=0.01)
        # Cast head and CNN params to float32 for stable optimization
        for m in [self.head, self.symbol_cnn, self.string_norm]:
            for param in m.parameters():
                if param.requires_grad:
                    param.data = param.data.float()
        if self.string_missing_emb.requires_grad:
            self.string_missing_emb.data = self.string_missing_emb.data.float()

    def forward(self, input_ids, attention_mask, pert_ids, string_feats, sym_ids):
        # AIDO.Cell-10M LoRA dual-pool features [B, 512]
        aido_feats = self.lora_adapter(input_ids, attention_mask, pert_ids)

        # STRING GNN PPI features: handle missing genes [B, 256]
        string_is_zero = (string_feats.abs().sum(dim=-1, keepdim=True) == 0)
        missing_emb = self.string_missing_emb.unsqueeze(0).expand(string_feats.shape[0], -1)
        string_feats = torch.where(string_is_zero, missing_emb.to(string_feats.dtype), string_feats)
        string_feats = self.string_norm(string_feats.float())

        # Character CNN features [B, 128]
        sym_feats = self.symbol_cnn(sym_ids)

        # Fuse all sources [B, 896]
        fused = torch.cat([aido_feats, string_feats, sym_feats], dim=-1)

        # Head: [B, 3*6640] → [B, 3, 6640]
        logits = self.head(fused)
        return logits.view(-1, N_CLASSES, N_GENES)


class DEGLightningModule(LightningModule):
    def __init__(self, lora_r: int = 4, lora_alpha: int = 8, lora_dropout: float = 0.1,
                 hidden_dim: int = 384, dropout: float = 0.4,
                 backbone_lr: float = 2e-4, head_lr: float = 6e-4,
                 weight_decay: float = 0.03,
                 gamma_focal: float = 2.0, label_smoothing: float = 0.05,
                 lr_patience: int = 5, lr_factor: float = 0.5, lr_min: float = 1e-6):
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
            self.model = DEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                hidden_dim=self.hparams.hidden_dim,
                dropout=self.hparams.dropout,
            )
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        if stage == "test":
            if self.trainer is not None and hasattr(self.trainer, "datamodule"):
                dm = self.trainer.datamodule
                if dm is not None and hasattr(dm, "test_pert_ids"):
                    self._test_pert_ids = dm.test_pert_ids
                    self._test_symbols = dm.test_symbols

    def _get_tokenizer_inputs(self, batch):
        """Tokenize pert_ids on-the-fly for LoRA forward pass."""
        pert_ids = batch["pert_id"]
        batch_data = [
            {"gene_ids": [pid.split(".")[0]], "expression": [1.0]}
            for pid in pert_ids
        ]
        # Tokenizer is stored in the LoRA adapter
        tokenizer = self.model.lora_adapter.tokenizer
        batch_inputs = tokenizer(batch_data, return_tensors="pt")
        input_ids = batch_inputs["input_ids"].to(self.device)
        attention_mask = batch_inputs["attention_mask"].to(self.device)
        return input_ids, attention_mask, pert_ids

    def _forward(self, batch):
        input_ids, attention_mask, pert_ids = self._get_tokenizer_inputs(batch)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pert_ids=pert_ids,
            string_feats=batch["string_feats"],
            sym_ids=batch["sym_ids"],
        )

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

        local_preds = torch.cat(self._val_preds, dim=0)
        local_labels = torch.cat(self._val_labels, dim=0)
        local_indices = torch.cat(self._val_indices, dim=0)
        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        # Gather all predictions from all ranks (F1 is non-decomposable, must gather first)
        all_preds = self.all_gather(local_preds)
        all_labels = self.all_gather(local_labels)
        all_indices = self.all_gather(local_indices)

        if self.trainer.is_global_zero:
            # De-duplicate by index
            preds = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
            labels = all_labels.view(-1, N_GENES).cpu().numpy()
            indices = all_indices.view(-1).cpu().numpy()

            unique_pos = np.unique(indices, return_index=True)[1]
            preds = preds[unique_pos]
            labels = labels[unique_pos]

            # Sort by original index to maintain order
            order = np.argsort(indices[unique_pos])
            preds = preds[order]
            labels = labels[order]

            f1 = compute_deg_f1(preds, labels)
            f1_tensor = torch.tensor(f1, dtype=torch.float32, device=self.device)
            self.log("val_f1", f1_tensor, prog_bar=True, sync_dist=True)  # sync_dist=True for EarlyStopping
        else:
            # Log a dummy tensor on non-zero ranks so EarlyStopping can access val_f1 on all ranks
            self.log("val_f1", torch.tensor(0.0, device=self.device), prog_bar=False, sync_dist=True)

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
            # Move to CPU first, then reshape to [N, 3, 6640]
            all_preds_cpu = all_preds.cpu()
            all_idx_cpu = all_idx.cpu()
            preds = all_preds_cpu.view(-1, N_CLASSES, N_GENES).numpy()
            idxs = all_idx_cpu.view(-1).numpy()

            unique_pos = np.unique(idxs, return_index=True)[1]
            preds = preds[unique_pos]
            sorted_idxs = idxs[unique_pos]

            order = np.argsort(sorted_idxs)
            preds = preds[order]
            final_idxs = sorted_idxs[order]

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "test_predictions.tsv"

            # DATA_ABSTRACT requires: prediction shape [3, 6640] (3 classes, 6640 genes)
            rows = []
            for rank_i, orig_i in enumerate(final_idxs):
                pred_2d = preds[rank_i]  # [3, 6640]
                rows.append({
                    "idx": self._test_pert_ids[int(orig_i)],
                    "input": self._test_symbols[int(orig_i)],
                    "prediction": json.dumps(pred_2d.tolist()),
                })

            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {out_path} ({len(rows)} rows)")

    def configure_optimizers(self):
        # Separate parameter groups: backbone LoRA (lower LR) vs head/CNN (higher LR)
        backbone_params = list(self.model.lora_adapter.backbone.parameters())
        head_params = list(self.model.head.parameters()) + \
                      list(self.model.symbol_cnn.parameters()) + \
                      list(self.model.string_norm.parameters()) + \
                      [self.model.string_missing_emb]

        # Filter to only trainable params
        backbone_trainable = [p for p in backbone_params if p.requires_grad]
        head_trainable = [p for p in head_params if p.requires_grad]

        param_groups = [
            {"params": backbone_trainable, "lr": self.hparams.backbone_lr,
             "weight_decay": self.hparams.weight_decay},
            {"params": head_trainable, "lr": self.hparams.head_lr,
             "weight_decay": self.hparams.weight_decay},
        ]

        opt = torch.optim.AdamW(param_groups)

        # ReduceLROnPlateau monitoring val_loss (stable signal, proven effective for node3-2)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",           # minimize val_loss
            patience=self.hparams.lr_patience,
            factor=self.hparams.lr_factor,
            min_lr=self.hparams.lr_min,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters (LoRA + head) to minimize checkpoint size."""
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
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%), plus {total_buffers} buffer values"
        )
        return trainable_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """Load from partial checkpoint (only trainable params saved)."""
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--backbone-lr", type=float, default=2e-4)
    p.add_argument("--head-lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--lr-patience", type=int, default=5)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--early-stopping-patience", type=int, default=15)
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

    fast_dev_run = args.fast_dev_run  # action="store_true" already gives boolean
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
        strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=600))

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-2-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1", mode="max", patience=args.early_stopping_patience, verbose=True
    )
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

    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        cache_dir=str(cache_dir),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        lr_min=args.lr_min,
    )

    trainer.fit(model_module, datamodule=datamodule)

    # Test phase: use single best checkpoint (no averaging - proven harmful)
    if args.fast_dev_run or args.debug_max_step is not None:
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
