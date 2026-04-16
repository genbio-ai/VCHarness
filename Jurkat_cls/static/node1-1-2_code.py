#!/usr/bin/env python3
"""
Node 1-1-2: AIDO.Cell-10M LoRA + Dual-Pool + STRING GNN PPI + Symbol CNN
=========================================================================
4-source multi-modal fusion for Jurkat DEG prediction.

Key improvements over parent (node1-1):
  - LoRA fine-tuning (r=4, all 8 layers) instead of frozen backbone
  - Per-gene dual pooling (gene_pos + mean_pool) instead of mean-pool only
  - STRING GNN PPI features (256-dim, frozen) for perturbed gene context
  - Character-level Symbol CNN (64-dim) for gene naming signal

Key improvements over sibling (node1-1-1):
  - AIDO.Cell-10M (256-dim, 8 layers) vs 100M — less overfitting on 1,500 samples
  - LoRA r=4 vs r=16 — 15× fewer backbone trainable params
  - ReduceLROnPlateau (adaptive) vs CosineAnnealingLR (fixed)

Architecture:
  AIDO.Cell-10M LoRA → dual pool → [B, 512]
  Frozen STRING GNN  → PPI embed → [B, 256]
  Symbol CNN         → char feat → [B,  64]
  Fusion: [B, 832] → LayerNorm → Linear(832,384) → GELU → Dropout → Linear(384, 3×6640)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import numpy as np
import pandas as pd

from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score as sk_f1_score

from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy

# ─── Constants ───────────────────────────────────────────────────────────────
N_GENES = 6_640
N_CLASSES = 3
AIDO_MODEL_PATH = "/home/Models/AIDO.Cell-10M"
STRING_GNN_PATH = "/home/Models/STRING_GNN"
AIDO_HIDDEN_DIM = 256   # AIDO.Cell-10M hidden dimension
GNN_HIDDEN_DIM = 256    # STRING GNN output dimension
SYMBOL_OUT_DIM = 64     # Symbol CNN output dimension
# Dual-pool (512) + GNN (256) + Symbol (64) = 832
FUSION_DIM = AIDO_HIDDEN_DIM * 2 + GNN_HIDDEN_DIM + SYMBOL_OUT_DIM

# Class weights: moderate minority-class boost (empirically validated in memory)
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)

# Character vocabulary for gene symbol CNN
_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-._/"
CHAR_VOCAB: Dict[str, int] = {c: i + 1 for i, c in enumerate(_CHARS)}
CHAR_VOCAB["<PAD>"] = 0
MAX_SYMBOL_LEN = 16
VOCAB_SIZE = len(CHAR_VOCAB) + 1  # +1 for safety


def encode_symbol(symbol: str, max_len: int = MAX_SYMBOL_LEN) -> torch.Tensor:
    """Encode a gene symbol string as a fixed-length character index tensor."""
    s = symbol.lower()[:max_len]
    chars = [CHAR_VOCAB.get(c, 0) for c in s]
    chars += [0] * (max_len - len(chars))  # right-pad with 0 (<PAD>)
    return torch.tensor(chars, dtype=torch.long)


# ─── Focal Loss ──────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weighting and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: [N, C],  targets: [N] int64"""
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(
            logits, targets, weight=w,
            label_smoothing=self.label_smoothing, reduction="none"
        )
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ─── Symbol CNN ──────────────────────────────────────────────────────────────
class SymbolCNN(nn.Module):
    """Character-level CNN encoder for gene symbol strings."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 32,
        out_dim: int = SYMBOL_OUT_DIM,
        max_len: int = MAX_SYMBOL_LEN,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        kernel_sizes = [3, 4, 5]
        n_filters = 16
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, k, padding=0)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, MAX_SYMBOL_LEN] long → [B, out_dim]"""
        emb = self.embedding(x)          # [B, L, embed_dim]
        emb = emb.transpose(1, 2)        # [B, embed_dim, L]
        pools = []
        for conv in self.convs:
            c = F.gelu(conv(emb))        # [B, n_filters, L-k+1]
            c = c.max(dim=2)[0]          # [B, n_filters]
            pools.append(c)
        cat = self.dropout(torch.cat(pools, dim=1))  # [B, n_filters * 3]
        return self.fc(cat)              # [B, out_dim]


# ─── Fusion Head ─────────────────────────────────────────────────────────────
class FusionHead(nn.Module):
    """Multi-modal fusion head: [B, fusion_dim] → [B, N_CLASSES * N_GENES]."""

    def __init__(
        self,
        in_dim: int = FUSION_DIM,
        hidden_dim: int = 384,
        out_dim: int = N_CLASSES * N_GENES,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Metric helper (mirrors calc_metric.py) ──────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """
    y_pred:          [N, 3, G]  float  (probabilities or logits)
    y_true_remapped: [N, G]     int    (0/1/2 after +1 remap from -1/0/1)
    Returns: per-gene macro F1 averaged over G genes (matches calc_metric.py).
    """
    n_genes = y_true_remapped.shape[1]
    f1_vals: List[float] = []
    y_hat = y_pred.argmax(axis=1)  # [N, G]
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp = y_hat[:, g]
        present = np.array([(yt == c).any() for c in range(3)])
        pf1 = sk_f1_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Dataset ─────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """Perturbation dataset with pre-tokenized AIDO.Cell inputs."""

    def __init__(
        self,
        df: pd.DataFrame,
        input_ids: torch.Tensor,      # [N, 19264] float32
        attention_mask: torch.Tensor, # [N, 19264] int64
        pert_pos: List[int],          # index in AIDO.Cell vocab (-1 if unknown)
        gnn_indices: List[int],       # index in STRING GNN nodes (-1 if unknown)
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.input_ids = input_ids            # [N, 19264]
        self.attention_mask = attention_mask  # [N, 19264]
        self.pert_pos = torch.tensor(pert_pos, dtype=torch.long)
        self.gnn_indices = torch.tensor(gnn_indices, dtype=torch.long)
        # Pre-encode gene symbols as character indices
        self.symbol_chars = torch.stack([encode_symbol(s) for s in self.symbols])  # [N, L]

        self.is_test = is_test
        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            self.labels = np.array(raw_labels, dtype=np.int8) + 1  # {-1,0,1} → {0,1,2}
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "input_ids": self.input_ids[idx],          # [19264] float32
            "attention_mask": self.attention_mask[idx], # [19264] int64
            "pert_pos": self.pert_pos[idx],             # scalar long
            "gnn_idx": self.gnn_indices[idx],           # scalar long
            "symbol_chars": self.symbol_chars[idx],     # [MAX_SYMBOL_LEN] long
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)  # [6640]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    # ------------------------------------------------------------------
    def _load_gnn_name_to_idx(self) -> Dict[str, int]:
        """Load STRING GNN node_names.json → {ensembl_id: node_index}."""
        node_names: List[str] = json.loads(
            (Path(STRING_GNN_PATH) / "node_names.json").read_text()
        )
        return {name: i for i, name in enumerate(node_names)}

    def _get_pert_pos(
        self, tokenizer: Any, pert_id: str
    ) -> int:
        """Get position of pert_id in AIDO.Cell 19264-gene vocabulary.
        Returns -1 if the gene ID is not in the vocabulary.
        """
        try:
            inputs = tokenizer(
                {"gene_ids": [pert_id], "expression": [1.0]},
                return_tensors="pt",
            )
            iids_raw = inputs["input_ids"]
            # Handle 1D [19264] or 2D [1, 19264] output
            iids = iids_raw[0] if iids_raw.ndim == 2 else iids_raw
            # Non-missing gene has expression > 0 (not the -1.0 placeholder)
            pos_candidates = (iids > -0.5).nonzero(as_tuple=True)[0]
            if len(pos_candidates) > 0:
                return int(pos_candidates[0])
        except Exception:
            pass
        return -1

    def _prepare_split(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        gnn_name_to_idx: Dict[str, int],
        split_name: str,
    ):
        """Pre-tokenize all samples and compute per-sample metadata."""
        pert_ids = df["pert_id"].tolist()
        N = len(pert_ids)
        print(f"[DataModule] Pre-tokenizing {N} samples for split='{split_name}'...")

        all_input_ids: List[torch.Tensor] = []
        all_attention_mask: List[torch.Tensor] = []
        all_pert_pos: List[int] = []
        all_gnn_idx: List[int] = []

        for i, pid in enumerate(pert_ids):
            # Tokenize single sample
            inputs = tokenizer(
                {"gene_ids": [pid], "expression": [1.0]},
                return_tensors="pt",
            )
            # AIDO.Cell tokenizer may return 1D [19264] or 2D [1, 19264] tensors
            # Normalize to 1D for consistent storage
            iids_raw = inputs["input_ids"]
            iids_1d = iids_raw[0] if iids_raw.ndim == 2 else iids_raw
            all_input_ids.append(iids_1d.float())  # [19264]

            attn_raw = inputs["attention_mask"]
            attn_1d = attn_raw[0] if attn_raw.ndim == 2 else attn_raw
            all_attention_mask.append(attn_1d.long())  # [19264]

            # Pert position in AIDO vocab (on 1D tensor)
            pos_candidates = (iids_1d > -0.5).nonzero(as_tuple=True)[0]
            pert_pos = int(pos_candidates[0]) if len(pos_candidates) > 0 else -1
            all_pert_pos.append(pert_pos)

            # STRING GNN index — strip any version suffix from Ensembl ID
            pid_clean = pid.split(".")[0]
            all_gnn_idx.append(gnn_name_to_idx.get(pid_clean, -1))

            if (i + 1) % 500 == 0 or (i + 1) == N:
                print(f"  {i + 1}/{N} done")

        input_ids_tensor = torch.stack(all_input_ids)       # [N, 19264]
        attention_mask_tensor = torch.stack(all_attention_mask)  # [N, 19264]
        return input_ids_tensor, attention_mask_tensor, all_pert_pos, all_gnn_idx

    def setup(self, stage: Optional[str] = None) -> None:
        # ── Tokenizer: rank 0 downloads first, barrier, all ranks load ──
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)

        # STRING GNN node name → index mapping
        gnn_name_to_idx = self._load_gnn_name_to_idx()

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")

            tr_iids, tr_attn, tr_pos, tr_gnn = self._prepare_split(
                train_df, tokenizer, gnn_name_to_idx, "train"
            )
            vl_iids, vl_attn, vl_pos, vl_gnn = self._prepare_split(
                val_df, tokenizer, gnn_name_to_idx, "val"
            )

            self.train_ds = PerturbationDataset(
                train_df, tr_iids, tr_attn, tr_pos, tr_gnn, is_test=False
            )
            self.val_ds = PerturbationDataset(
                val_df, vl_iids, vl_attn, vl_pos, vl_gnn, is_test=False
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            ts_iids, ts_attn, ts_pos, ts_gnn = self._prepare_split(
                test_df, tokenizer, gnn_name_to_idx, "test"
            )
            self.test_ds = PerturbationDataset(
                test_df, ts_iids, ts_attn, ts_pos, ts_gnn, is_test=True
            )
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ─── LightningModule ──────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):

    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        hidden_dim: int = 384,
        dropout: float = 0.4,
        lr_backbone: float = 2e-4,
        lr_head: float = 6e-4,
        weight_decay: float = 0.03,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        lr_patience: int = 8,
        lr_factor: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialized in setup()
        self.backbone: Optional[nn.Module] = None
        self.symbol_encoder: Optional[SymbolCNN] = None
        self.head: Optional[FusionHead] = None
        self.criterion: Optional[FocalLoss] = None

        # Accumulators
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    # ── Model initialization (called once per rank) ────────────────────────
    def setup(self, stage: Optional[str] = None) -> None:
        if self.backbone is not None:
            # Avoid re-initializing when setup is called multiple times
            if stage == "test" and hasattr(self.trainer, "datamodule"):
                dm = self.trainer.datamodule
                if hasattr(dm, "test_pert_ids"):
                    self._test_pert_ids = dm.test_pert_ids
                    self._test_symbols = dm.test_symbols
            return

        # ── 1. AIDO.Cell-10M + LoRA ────────────────────────────────────────
        backbone = AutoModel.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,  # fine-tune all 8 transformer layers
        )
        backbone = get_peft_model(backbone, lora_cfg)
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Cast LoRA params to float32 for stable mixed-precision training
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()
        self.backbone = backbone

        # ── 2. Frozen STRING GNN node embeddings ──────────────────────────
        gnn_model = AutoModel.from_pretrained(STRING_GNN_PATH, trust_remote_code=True)
        gnn_model.eval()
        graph = torch.load(Path(STRING_GNN_PATH) / "graph_data.pt", weights_only=False)
        edge_index: torch.Tensor = graph["edge_index"]        # LongTensor [2, E]
        edge_weight = graph.get("edge_weight", None)           # FloatTensor [E] or None

        with torch.no_grad():
            gnn_out = gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            gnn_emb = gnn_out.last_hidden_state.float().detach()  # [18870, 256], CPU
        self.register_buffer("gnn_node_emb", gnn_emb)
        del gnn_model
        torch.cuda.empty_cache()

        # ── 3. Symbol CNN ─────────────────────────────────────────────────
        self.symbol_encoder = SymbolCNN(
            vocab_size=VOCAB_SIZE,
            embed_dim=32,
            out_dim=SYMBOL_OUT_DIM,
            max_len=MAX_SYMBOL_LEN,
        )

        # ── 4. Fusion Head ────────────────────────────────────────────────
        self.head = FusionHead(
            in_dim=FUSION_DIM,
            hidden_dim=self.hparams.hidden_dim,
            out_dim=N_CLASSES * N_GENES,
            dropout=self.hparams.dropout,
        )

        # ── 5. Focal Loss ─────────────────────────────────────────────────
        self.criterion = FocalLoss(
            gamma=self.hparams.gamma_focal,
            weight=CLASS_WEIGHTS,
            label_smoothing=self.hparams.label_smoothing,
        )

        if stage == "test" and hasattr(self.trainer, "datamodule"):
            dm = self.trainer.datamodule
            if hasattr(dm, "test_pert_ids"):
                self._test_pert_ids = dm.test_pert_ids
                self._test_symbols = dm.test_symbols

    # ── Forward pass ──────────────────────────────────────────────────────
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        input_ids = batch["input_ids"]       # [B, 19264] float32
        attention_mask = batch["attention_mask"]  # [B, 19264] int64
        pert_pos = batch["pert_pos"]         # [B] long
        gnn_idx = batch["gnn_idx"]           # [B] long
        symbol_chars = batch["symbol_chars"] # [B, MAX_SYMBOL_LEN] long
        B = input_ids.shape[0]

        # ── AIDO.Cell-10M backbone (LoRA fine-tuned) ─────────────────────
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        hidden = outputs.last_hidden_state.float()  # [B, 19266, 256]

        # Dual pooling: per-gene positional embedding + global mean-pool
        mean_pool = hidden[:, :19264, :].mean(dim=1)  # [B, 256]
        gene_emb = mean_pool.clone()                   # fallback for unknown genes
        valid_pos = pert_pos >= 0
        if valid_pos.any():
            gene_emb[valid_pos] = hidden[valid_pos, pert_pos[valid_pos], :]
        aido_features = torch.cat([gene_emb, mean_pool], dim=1)  # [B, 512]

        # ── Frozen STRING GNN features ────────────────────────────────────
        gnn_feats = torch.zeros(B, GNN_HIDDEN_DIM, device=self.device)
        valid_gnn = gnn_idx >= 0
        if valid_gnn.any():
            gnn_feats[valid_gnn] = self.gnn_node_emb[gnn_idx[valid_gnn]]

        # ── Symbol CNN features ───────────────────────────────────────────
        symbol_feats = self.symbol_encoder(symbol_chars)  # [B, 64]

        # ── 4-source fusion → prediction head ────────────────────────────
        fused = torch.cat([aido_features, gnn_feats, symbol_feats], dim=1)  # [B, 832]
        logits = self.head(fused)                          # [B, 3 * N_GENES]
        return logits.view(B, N_CLASSES, N_GENES)          # [B, 3, 6640]

    # ── Loss helper ───────────────────────────────────────────────────────
    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """logits: [B, 3, G],  labels: [B, G] int64"""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
        labels_flat = labels.reshape(-1)                       # [B*G]
        return self.criterion(logits_flat, labels_flat)

    # ── Training / Validation / Test steps ───────────────────────────────
    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, G]
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, dim=0)    # [N_local, 3, G]
        local_labels = torch.cat(self._val_labels, dim=0)  # [N_local, G]

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        # Compute per-rank F1, then average across ranks (consistent with parent)
        local_f1 = compute_deg_f1(
            local_preds.numpy(), local_labels.numpy()
        )
        world_size = self.trainer.world_size if self.trainer.world_size else 1
        if world_size > 1:
            import torch.distributed as dist
            f1_t = torch.tensor(local_f1, dtype=torch.float32, device=self.device)
            dist.all_reduce(f1_t, op=dist.ReduceOp.SUM)
            f1 = (f1_t / world_size).item()
        else:
            f1 = local_f1

        self.log("val_f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, G]
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, dim=0)  # [N_local, 3, G]
        local_idx = torch.cat(self._test_indices, dim=0)  # [N_local]

        # Gather across all ranks
        all_preds = self.all_gather(local_preds)  # [world_size, N_local, 3, G]
        all_idx = self.all_gather(local_idx)       # [world_size, N_local]

        self._test_preds.clear()
        self._test_indices.clear()

        if self.trainer.is_global_zero:
            preds = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
            idxs = all_idx.view(-1).cpu().numpy()

            # Deduplicate (DDP may pad the last batch)
            _, unique_pos = np.unique(idxs, return_index=True)
            preds = preds[unique_pos]
            sorted_idxs = idxs[unique_pos]

            # Sort by original order
            order = np.argsort(sorted_idxs)
            preds = preds[order]
            final_idxs = sorted_idxs[order]

            # Write test_predictions.tsv
            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "test_predictions.tsv"

            rows = []
            for rank_i, orig_i in enumerate(final_idxs):
                rows.append({
                    "idx": self._test_pert_ids[orig_i],
                    "input": self._test_symbols[orig_i],
                    "prediction": json.dumps(preds[rank_i].tolist()),
                })
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {out_path}")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    def configure_optimizers(self):
        backbone_params = [
            p for p in self.backbone.parameters() if p.requires_grad
        ]
        head_params = (
            list(self.symbol_encoder.parameters())
            + list(self.head.parameters())
        )
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_params,
                    "lr": self.hparams.lr_backbone,
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": head_params,
                    "lr": self.hparams.lr_head,
                    "weight_decay": self.hparams.weight_decay,
                },
            ]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=self.hparams.lr_patience,
            factor=self.hparams.lr_factor,
            min_lr=1e-6,
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

    # ── Checkpoint: save only trainable params + buffers ─────────────────
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {
            prefix + n for n, p in self.named_parameters() if p.requires_grad
        }
        # Exclude the large deterministic GNN buffer to save checkpoint space
        buffer_keys_to_save = {
            prefix + n
            for n, _ in self.named_buffers()
            if "gnn_node_emb" not in n  # recomputed in setup()
        }
        keep = {
            k: v
            for k, v in full_sd.items()
            if k in trainable_keys or k in buffer_keys_to_save
        }
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} trainable params "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
        return keep

    def load_state_dict(self, state_dict, strict=True):
        missing, unexpected = [], []
        # Use strict=False so missing gnn_node_emb buffer doesn't error
        result = super().load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            # Filter out gnn_node_emb (expected to be missing)
            true_missing = [k for k in result.missing_keys if "gnn_node_emb" not in k]
            if true_missing:
                self.print(f"Warning: Missing checkpoint keys: {true_missing[:5]}")
        if result.unexpected_keys:
            self.print(f"Warning: Unexpected checkpoint keys: {result.unexpected_keys[:5]}")
        return result


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node 1-1-2: AIDO.Cell-10M LoRA + STRING GNN + Symbol CNN"
    )
    # Data / output
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-workers", type=int, default=4)
    # Batch sizes
    parser.add_argument("--micro-batch-size", type=int, default=4,
                        help="Batch size per GPU. AIDO.Cell-10M LoRA ~3.21 GiB/GPU at BS=1.")
    parser.add_argument("--global-batch-size", type=int, default=32,
                        help="Effective batch size. Must be multiple of micro_batch_size * 8.")
    # Training
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    # LoRA
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    # Head
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.4)
    # Optimizer
    parser.add_argument("--lr-backbone", type=float, default=2e-4)
    parser.add_argument("--lr-head", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    # Scheduler + Early stopping
    parser.add_argument("--lr-patience", type=int, default=8)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    # Loss
    parser.add_argument("--gamma-focal", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    # Debug
    parser.add_argument("--debug-max-step", type=int, default=None,
                        help="Limit global training steps for quick debugging.")
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Run 1 batch for train/val/test (smoke test).")
    args = parser.parse_args()

    # ── Output dir ────────────────────────────────────────────────────────
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── DDP / GPU setup ───────────────────────────────────────────────────
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    if n_gpus == 0:
        n_gpus = 1

    # ── Trainer limits ────────────────────────────────────────────────────
    max_steps = -1
    fast_dev_run: Any = args.fast_dev_run
    limit_train_batches: Any = 1.0
    limit_val_batches: Any = 1.0
    limit_test_batches: Any = 1.0
    val_check_interval: Any = args.val_check_interval

    if args.debug_max_step is not None:
        max_steps = args.debug_max_step
        limit_val_batches = 1
        limit_test_batches = 1
        val_check_interval = 1.0

    accumulate_grad_batches = max(
        1, args.global_batch_size // (args.micro_batch_size * n_gpus)
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=10)
    # Only enable EarlyStopping when val_f1 will be available (not in debug mode)
    callbacks = [checkpoint_callback, lr_monitor, progress_bar]
    if args.debug_max_step is None and not args.fast_dev_run:
        early_stop_callback = EarlyStopping(
            monitor="val_f1",
            mode="max",
            patience=args.early_stopping_patience,
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    # ── Loggers ───────────────────────────────────────────────────────────
    csv_logger = CSVLogger(
        save_dir=str(output_dir / "logs"), name="csv_logs"
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    # ── Strategy ──────────────────────────────────────────────────────────
    if n_gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=True,  # some backbone params have no grads (non-LoRA)
            timeout=timedelta(seconds=300),
        )
    else:
        strategy = SingleDeviceStrategy(device="cuda:0")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad_batches,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=(
            val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=callbacks,
        logger=[csv_logger, tensorboard_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    # ── DataModule and Model ──────────────────────────────────────────────
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model = DEGLightningModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    trainer.fit(model, datamodule=datamodule)

    # ── Test ──────────────────────────────────────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    if test_results:
        print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
