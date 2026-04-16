#!/usr/bin/env python3
"""
Node 1-3-1-1-1-1-1-1-1-1: Regularization Calibration + Cosine LR + Stronger Class Weights
============================================================================================
Key changes from parent node1-3-1-1-1-1-1-1-1 (test F1=0.4480, regression vs parent 0.4490):

ROOT CAUSE ANALYSIS:
The parent's over-regularization (dropout=0.55, weight_decay=0.05) caused underfitting.
Evidence:
  - Parent test F1: 0.4480 (-0.0010 regression from grandparent's 0.4490)
  - val_loss peaked at 0.62 vs grandparent's 0.68 — regularization was effective at
    reducing val_loss divergence but at the cost of lower peak val_f1
  - The 0.55 dropout constrained model capacity below the optimal operating point on the
    bias-variance tradeoff curve
  - The model couldn't improve after epoch 23, suggesting it was "stuck" in a narrow band

THIS NODE'S TARGETED FIXES:
1. REDUCE DROPOUT: 0.55 → 0.45
   - Feedback priority 1: "reduce dropout from 0.55 to 0.45-0.48"
   - 0.45 is a compromise between grandparent's 0.4 (insufficient) and parent's 0.55 (excessive)
   - Should restore model capacity while still providing meaningful regularization

2. REDUCE WEIGHT DECAY: 0.05 → 0.03
   - Feedback priority 2: "revert weight_decay to 0.03 (proven value)"
   - 0.03 is proven across tree-best nodes (node3-2, node2-3-1-1-1-1 both use 0.03)
   - Combined with dropout reduction, brings regularization to a healthier level

3. SWITCH TO COSINE ANNEALING LR SCHEDULE: ReduceLROnPlateau → CosineAnnealingLR
   - Feedback priority 3: "Replace ReduceLROnPlateau with CosineAnnealingLR"
   - The model converged rapidly (main learning by epoch 8) then stagnated at epoch 23
   - Cosine schedule provides smoother, more structured LR decay to potentially help
     escape local optima during the later training phase
   - T_max=50 with eta_min=1e-6 covers the expected training duration before early stop
   - This is a distinct optimization approach from the parent's ReduceLROnPlateau

4. STRENGTHEN CLASS WEIGHTS: [5.0, 1.0, 10.0] → [6.0, 1.0, 12.0]
   - Tree-best node2-2-3-1-1 (F1=0.4625) uses [6.0, 1.0, 12.0] class weights
   - These stronger weights put more emphasis on minority classes (-1 and +1)
   - The 20% increase in minority class weighting may help the model better learn
     the rare but important differential expression classes

ALL OTHER HYPERPARAMETERS UNCHANGED:
- CNN dim=64 (proven alignment with tree-best node3-2 architecture)
- FUSION_DIM=832 (512 + 256 + 64 = 832)
- backbone_lr=2e-4 (proven tree-best value)
- head_lr=6e-4 (3x backbone ratio)
- LoRA r=4, alpha=8, lora_dropout=0.1, all 8 QKV layers (~18K trainable params)
- FocalLoss(γ=2.0, label_smoothing=0.05)
- String-keyed val_f1 deduplication (RETAINED — zero val-test gap confirmed)
- max_epochs=100, early_stopping_patience=20
- seed=0 (matches tree-best node3-2 default seed)

EXPECTED IMPROVEMENT:
- Regularization calibration (dropout 0.55→0.45, wd 0.05→0.03): ~+0.003-0.010 F1
- Cosine LR schedule: ~+0.003-0.008 F1 (potentially helps escape epoch 23 stagnation)
- Stronger class weights [6,1,12]: ~+0.001-0.005 F1 (better minority class learning)
- Combined target F1: 0.458-0.472
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

# KEY CHANGE: Stronger class weights [6.0, 1.0, 12.0] (tree-best node2-2-3-1-1 uses this)
# Parent used [5.0, 1.0, 10.0]; this 20% increase may improve minority class learning
# Class mapping: 0=down-regulated(-1), 1=unchanged(0), 2=up-regulated(+1)
CLASS_WEIGHTS = torch.tensor([6.0, 1.0, 12.0], dtype=torch.float32)

CHAR_VOCAB = {c: i + 1 for i, c in enumerate(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.abcdefghijklmnopqrstuvwxyz"
)}
CHAR_VOCAB["<pad>"] = 0
MAX_SYM_LEN = 16

AIDO_HIDDEN_DIM = 256   # AIDO.Cell-10M hidden size
STRING_HIDDEN_DIM = 256

# CNN dim = 64 (matching tree-best node3-2 and node2-3 lineage — UNCHANGED from parent)
# Parent confirmed this is the correct architecture alignment
CHAR_CNN_DIM = 64       # 2 kernels x 32 output each = 64-dim (matching tree-best)

# Total fusion dimension: 512 (AIDO dual-pool) + 256 (STRING) + 64 (Char CNN) = 832
FUSION_DIM = AIDO_HIDDEN_DIM * 2 + STRING_HIDDEN_DIM + CHAR_CNN_DIM  # 832


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        log_probs = F.log_softmax(logits, dim=1)
        ce = F.cross_entropy(logits, targets, weight=w,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1))
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
    Only STRING GNN features (which remain fully frozen) are cached.
    AIDO features are computed live during training via the LoRA backbone.
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
    print(f"[PreCompute] STRING GNN feature cache saved -> {cache_path}")

    del string_model
    torch.cuda.empty_cache()

    return results


class DEGDataset(Dataset):
    """
    Dataset for DEG prediction with LoRA fine-tuning.
    Stores pert_id strings for live tokenization (required for LoRA gradient flow).
    STRING GNN PPI features are pre-computed and stored (frozen).
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
    """
    Multi-scale character-level CNN for gene symbol encoding.

    Uses out_dim=32 (total output: 32+32=64 dim) — matching the tree-best node3-2
    and node2-3 lineage (both use 64-dim CNN output).
    Unchanged from parent node1-3-1-1-1-1-1-1-1.
    """
    def __init__(self, vocab_size, embed_dim=32, out_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embed_dim, out_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, out_dim, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(out_dim * 2)
        self.out_dim = out_dim * 2  # 64-dim (32+32)

    def forward(self, ids):
        x = self.embedding(ids).transpose(1, 2)
        f3 = F.gelu(self.conv3(x)).max(dim=-1).values
        f5 = F.gelu(self.conv5(x)).max(dim=-1).values
        return self.norm(torch.cat([f3, f5], dim=-1))


class LoRAAdapter(nn.Module):
    """
    AIDO.Cell-10M with LoRA fine-tuning (r=4, all 8 layers, QKV projections).

    Identical to parent node1-3-1-1-1-1-1-1-1 — proven tree-best configuration.
    - r=4 (very low rank): ~18K trainable params, minimal overfitting risk
    - All 8 transformer layers: broader adaptation capacity
    - alpha=8 (2x r): standard scaling factor
    - lora_dropout=0.1: light regularization on LoRA adaptation paths
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

        # Build gene_id_to_pos lookup table
        self.gene_id_to_pos: Dict[str, int] = {}
        if hasattr(self.tokenizer, "gene_id_to_index"):
            self.gene_id_to_pos = {k: int(v) for k, v in self.tokenizer.gene_id_to_index.items()}
        elif hasattr(self.tokenizer, "gene_to_index"):
            self.gene_id_to_pos = {k: int(v) for k, v in self.tokenizer.gene_to_index.items()}

        base_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)

        # Disable FlashAttention when using LoRA:
        # AIDO.Cell's flash_self module bypasses LoRA modifications to attention weights
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

        # Cast model to bf16 for memory efficiency
        for name, param in self.backbone.named_parameters():
            orig_requires_grad = param.requires_grad
            param.data = param.data.to(dtype=torch.bfloat16)
            param.requires_grad = orig_requires_grad

        # Cast LoRA params back to float32 for stable optimization
        # Mixed precision: frozen backbone in bf16, trainable LoRA in float32
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

        Returns:
            dual_pool: [B, 512] = concat(gene_pos_emb[B,256], mean_emb[B,256])
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
    CNN dim = 64 → FUSION_DIM = 832 (matching tree-best node3-2 architecture).

    Architecture (total fusion dim = 832):
    - AIDO.Cell-10M LoRA (512-dim): dual pool = gene_pos_emb(256) + mean(256)
    - STRING GNN PPI (256-dim): frozen pre-computed topology embeddings
    - Character-level CNN (64-dim): multi-scale gene symbol encoder
    - MLP head: LayerNorm(832) -> Linear(384) -> GELU -> Dropout(0.45) -> Linear(19920)
      KEY CHANGE: Dropout reduced from 0.55 → 0.45 (addresses over-regularization)
    """
    def __init__(self, lora_r: int = 4, lora_alpha: int = 8, lora_dropout: float = 0.1,
                 hidden_dim: int = 384, dropout: float = 0.45):
        super().__init__()
        self.lora_adapter = LoRAAdapter(
            model_dir=AIDO_MODEL_DIR,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        # out_dim=32 (total 64-dim) matching tree-best node3-2 configuration
        self.symbol_cnn = SymbolCNN(vocab_size=len(CHAR_VOCAB), embed_dim=32, out_dim=32)
        self.string_missing_emb = nn.Parameter(torch.zeros(STRING_HIDDEN_DIM))
        self.string_norm = nn.LayerNorm(STRING_HIDDEN_DIM)

        # MLP head: 832 -> 384 -> 19920 (N_CLASSES * N_GENES = 3 * 6640)
        # KEY CHANGE: Dropout reduced from 0.55 to 0.45 (addresses parent's over-regularization)
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Linear(FUSION_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, N_CLASSES * N_GENES),
        )
        self._init_weights()

    def _init_weights(self):
        for m in [self.head, self.symbol_cnn]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.normal_(self.string_missing_emb, mean=0.0, std=0.01)
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

        # Character CNN features [B, 64]
        sym_feats = self.symbol_cnn(sym_ids)

        # Fuse all sources [B, 832]
        fused = torch.cat([aido_feats, string_feats, sym_feats], dim=-1)

        # Head: [B, 3*6640] -> [B, 3, 6640]
        logits = self.head(fused)
        return logits.view(-1, N_CLASSES, N_GENES)


class DEGLightningModule(LightningModule):
    def __init__(self, lora_r: int = 4, lora_alpha: int = 8, lora_dropout: float = 0.1,
                 hidden_dim: int = 384, dropout: float = 0.45,
                 backbone_lr: float = 2e-4, head_lr: float = 6e-4,
                 weight_decay: float = 0.03,
                 gamma_focal: float = 2.0, label_smoothing: float = 0.05,
                 max_epochs: int = 100,
                 cosine_t_max: int = 50, cosine_eta_min: float = 1e-6):
        super().__init__()
        self.save_hyperparameters()
        self.model = None
        self.criterion = None
        # RETAINED: string-keyed deduplication from ancestors (confirmed working, zero val-test gap)
        self._val_preds_dict: Dict[str, np.ndarray] = {}
        self._val_labels_dict: Dict[str, np.ndarray] = {}
        self._test_preds_dict: Dict[str, np.ndarray] = {}
        self._test_symbols_dict: Dict[str, str] = {}
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

        # CRITICAL (retained from ancestors): string-keyed deduplication avoids DDP integer
        # index collision. Using unique pert_id strings guarantees correct deduplication
        # regardless of DDP rank assignment or padding.
        probs = F.softmax(logits.detach().float(), dim=1).cpu().numpy()  # [B, 3, N_GENES]
        labels_np = batch["label"].cpu().numpy()  # [B, N_GENES]
        pert_ids = batch["pert_id"]  # list of strings

        for i, pid in enumerate(pert_ids):
            self._val_preds_dict[pid] = probs[i]
            self._val_labels_dict[pid] = labels_np[i]

    def on_validation_epoch_end(self):
        if not self._val_preds_dict:
            return

        # Convert local dict to lists for gathering
        local_pids = list(self._val_preds_dict.keys())
        local_preds = np.stack([self._val_preds_dict[p] for p in local_pids])   # [N_local, 3, N_GENES]
        local_labels = np.stack([self._val_labels_dict[p] for p in local_pids]) # [N_local, N_GENES]

        # Clear local dicts
        self._val_preds_dict.clear()
        self._val_labels_dict.clear()

        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

        if is_distributed:
            preds_tensor = torch.from_numpy(local_preds).to(self.device)  # [N_local, 3, N_GENES]
            labels_tensor = torch.from_numpy(local_labels).to(self.device)  # [N_local, N_GENES]

            # Encode pert_id strings as UTF-8 byte tensors for all_gather
            MAX_STR_LEN = 64  # Ensembl gene IDs are ~15 chars, 64 is safe
            str_bytes_list = [p.encode("utf-8")[:MAX_STR_LEN] for p in local_pids]
            str_lens = torch.tensor([len(b) for b in str_bytes_list], dtype=torch.int32, device=self.device)
            max_len = str_lens.max().item()

            str_padded = torch.zeros(len(local_pids), max_len, dtype=torch.uint8, device=self.device)
            for i, b in enumerate(str_padded):
                str_padded[i, :len(str_bytes_list[i])] = torch.tensor(
                    list(str_bytes_list[i]), dtype=torch.uint8, device=self.device
                )

            all_preds = self.all_gather(preds_tensor)       # [world_size, N_local, 3, N_GENES]
            all_labels = self.all_gather(labels_tensor)     # [world_size, N_local, N_GENES]
            all_str_lens = self.all_gather(str_lens)        # [world_size, N_local]
            all_str_padded = self.all_gather(str_padded)    # [world_size, N_local, max_len]
        else:
            # Single GPU: no gathering needed
            all_preds = torch.from_numpy(local_preds).unsqueeze(0)
            all_labels = torch.from_numpy(local_labels).unsqueeze(0)
            str_bytes_list = [p.encode("utf-8")[:64] for p in local_pids]
            all_str_padded = torch.zeros(1, len(local_pids), 64, dtype=torch.uint8)
            for i, b in enumerate(str_bytes_list):
                all_str_padded[0, i, :len(b)] = torch.tensor(list(b), dtype=torch.uint8)
            all_str_lens = torch.tensor([[len(b) for b in str_bytes_list]], dtype=torch.int32)

        # All ranks decode pert_id strings and compute F1 (data is identical after all_gather)
        world_size = all_preds.shape[0]
        n_local = all_preds.shape[1]
        all_pids: List[str] = []
        for r in range(world_size):
            for i in range(n_local):
                length = all_str_lens[r, i].item()
                if length > 0:
                    pid_bytes = all_str_padded[r, i, :length].cpu().numpy().tobytes().rstrip(b'\x00')
                    all_pids.append(pid_bytes.decode("utf-8"))

        # Flatten across world_size dimension
        flat_preds = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
        flat_labels = all_labels.view(-1, N_GENES).cpu().numpy()

        # De-duplicate by pert_id string (guaranteed unique by data spec)
        seen_pids: set = set()
        dedup_preds: List[np.ndarray] = []
        dedup_labels: List[np.ndarray] = []
        n_actual = len(all_pids)
        for i in range(n_actual):
            pid = all_pids[i]
            if pid not in seen_pids:
                seen_pids.add(pid)
                dedup_preds.append(flat_preds[i])
                dedup_labels.append(flat_labels[i])

        preds = np.stack(dedup_preds)       # [N_val, 3, N_GENES]
        labels_arr = np.stack(dedup_labels) # [N_val, N_GENES]

        f1 = compute_deg_f1(preds, labels_arr)
        self.print(f"  [Val] de-duped samples: {len(dedup_preds)}, val_f1={f1:.4f}")
        f1_tensor = torch.tensor(f1, dtype=torch.float32, device=self.device)
        # sync_dist=True is safe here because all ranks have identical f1 values after
        # string-keyed deduplication (same input data). Using sync_dist=True suppresses
        # the Lightning warning and ensures consistent logging across ranks.
        self.log("val_f1", f1_tensor, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        logits = self._forward(batch)
        probs = F.softmax(logits.detach().float(), dim=1).cpu().numpy()  # [B, 3, N_GENES]
        pert_ids = batch["pert_id"]  # list of strings
        indices = batch["idx"].cpu().numpy()  # original dataset indices

        for i, pid in enumerate(pert_ids):
            self._test_preds_dict[pid] = probs[i]
            # Store symbol using original index
            orig_idx = int(indices[i])
            if orig_idx < len(self._test_symbols):
                self._test_symbols_dict[pid] = self._test_symbols[orig_idx]

    def on_test_epoch_end(self):
        if not self._test_preds_dict:
            return

        local_pids = list(self._test_preds_dict.keys())
        local_preds = np.stack([self._test_preds_dict[p] for p in local_pids])  # [N_local, 3, N_GENES]

        # Gather all predictions from all ranks using string keys (no hash collisions)
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if is_distributed:
            preds_tensor = torch.from_numpy(local_preds).to(self.device)

            MAX_STR_LEN = 64
            str_bytes_list = [p.encode("utf-8")[:MAX_STR_LEN] for p in local_pids]
            str_lens = torch.tensor([len(b) for b in str_bytes_list], dtype=torch.int32, device=self.device)
            max_len = str_lens.max().item()

            str_padded = torch.zeros(len(local_pids), max(1, max_len), dtype=torch.uint8, device=self.device)
            for i, b in enumerate(str_bytes_list):
                if len(b) > 0:
                    str_padded[i, :len(b)] = torch.tensor(list(b), dtype=torch.uint8, device=self.device)

            all_preds = self.all_gather(preds_tensor)
            all_str_lens = self.all_gather(str_lens)
            all_str_padded = self.all_gather(str_padded)
        else:
            all_preds = torch.from_numpy(local_preds).unsqueeze(0)
            str_bytes_list = [p.encode("utf-8")[:64] for p in local_pids]
            all_str_padded = torch.zeros(1, len(local_pids), 64, dtype=torch.uint8)
            for i, b in enumerate(str_bytes_list):
                all_str_padded[0, i, :len(b)] = torch.tensor(list(b), dtype=torch.uint8)
            all_str_lens = torch.tensor([[len(b) for b in str_bytes_list]], dtype=torch.int32)

        self._test_preds_dict.clear()

        # All ranks have identical gathered data (all_gather produces same tensors on all ranks).
        # Use global_rank check to avoid redundant file writes from all ranks.
        is_rank_zero = (torch.distributed.is_initialized() and
                        torch.distributed.get_rank() == 0) or \
                       not torch.distributed.is_initialized()
        if is_rank_zero:
            # Decode pert_id strings
            world_size = all_preds.shape[0]
            n_local = all_preds.shape[1]
            all_pids: List[str] = []
            for r in range(world_size):
                for i in range(n_local):
                    length = all_str_lens[r, i].item()
                    if length > 0:
                        pid_bytes = all_str_padded[r, i, :length].cpu().numpy().tobytes().rstrip(b'\x00')
                        all_pids.append(pid_bytes.decode("utf-8"))

            # Flatten predictions
            flat_preds = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()

            # De-duplicate by pert_id string
            pid_to_pred: Dict[str, np.ndarray] = {}
            n_actual = len(all_pids)
            for i in range(n_actual):
                pid = all_pids[i]
                if pid not in pid_to_pred:
                    pid_to_pred[pid] = flat_preds[i]

            # Build ordered output matching test_pert_ids order
            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "test_predictions.tsv"

            rows = []
            for pid in self._test_pert_ids:
                if pid in pid_to_pred:
                    pred_2d = pid_to_pred[pid]  # [3, N_GENES]
                    sym = self._test_symbols_dict.get(pid, "")
                    rows.append({
                        "idx": pid,
                        "input": sym,
                        "prediction": json.dumps(pred_2d.tolist()),
                    })

            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved -> {out_path} ({len(rows)} rows)")

    def configure_optimizers(self):
        # Separate parameter groups: backbone LoRA (lower LR) vs head/CNN (higher LR)
        backbone_params = list(self.model.lora_adapter.backbone.parameters())
        head_params = (
            list(self.model.head.parameters())
            + list(self.model.symbol_cnn.parameters())
            + list(self.model.string_norm.parameters())
            + [self.model.string_missing_emb]
        )

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

        # KEY CHANGE: Switch from ReduceLROnPlateau to CosineAnnealingLR
        #
        # Rationale:
        # - Parent converged rapidly (main learning by epoch 8) then stagnated at epoch 23
        # - ReduceLROnPlateau with patience=30 effectively never fired (as designed),
        #   meaning LR stayed constant at 2e-4 throughout — but this may have contributed
        #   to the stagnation after the epoch 23 peak
        # - Cosine annealing provides smooth, structured LR decay that can help the model
        #   explore different optimization trajectories during the later training phase
        # - T_max=50: covers the expected training window (early stop triggers ~40-50 epochs)
        # - eta_min=1e-6: floor to prevent LR from reaching zero too early
        # - Unlike ReduceLROnPlateau, cosine schedule is deterministic and not affected
        #   by metric oscillation — it provides a stable optimization landscape
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.hparams.cosine_t_max,
            eta_min=self.hparams.cosine_eta_min,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
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
        self.print(
            f"Loading checkpoint: {loaded_trainable} trainable params and {loaded_buffers} buffers"
        )
        return super().load_state_dict(state_dict, strict=False)


def parse_args():
    p = argparse.ArgumentParser(
        description="Node 1-3-1-1-1-1-1-1-1-1: 4-source LoRA DEG predictor (832-dim, dropout=0.45, cosine LR)"
    )
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=100)
    # Proven tree-best backbone and head LR (unchanged from parent)
    p.add_argument("--backbone-lr", type=float, default=2e-4)
    p.add_argument("--head-lr", type=float, default=6e-4)
    # KEY CHANGE: weight_decay reverted 0.05 → 0.03 (proven value; parent's 0.05 caused over-regularization)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--hidden-dim", type=int, default=384)
    # KEY CHANGE: dropout reverted 0.55 → 0.45 (compromise; parent's 0.55 caused underfitting)
    p.add_argument("--dropout", type=float, default=0.45)
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    # KEY CHANGE: Cosine LR parameters (replacing ReduceLROnPlateau)
    p.add_argument("--cosine-t-max", type=int, default=50)
    p.add_argument("--cosine-eta-min", type=float, default=1e-6)
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    return p.parse_args()


def main():
    # seed=0 matches tree-best node3-2 (default seed) — retained from parent
    pl.seed_everything(0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Cache directory is the same as output_dir — feature_cache.npz is saved directly under run/
    cache_dir = output_dir

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
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
        filename="node1-3-1-1-1-1-1-1-1-1-{epoch:03d}-{val_f1:.4f}",
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
        max_epochs=args.max_epochs,
        cosine_t_max=args.cosine_t_max,
        cosine_eta_min=args.cosine_eta_min,
    )

    trainer.fit(model_module, datamodule=datamodule)

    # Test phase: use single best checkpoint
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
