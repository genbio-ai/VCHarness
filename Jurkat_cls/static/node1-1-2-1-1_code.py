#!/usr/bin/env python3
"""
Node 1-2: Cross-Attention Fusion DEG Predictor (Stabilized)
=============================================================
AIDO.Cell-10M LoRA + STRING GNN PPI + Symbol CNN, fused through a multi-head
3-layer TransformerEncoder over 4 feature tokens, followed by a deeper
256→384→19920 prediction head trained with focal loss (NO manifold mixup).

Key improvements over parent (node1-1-2-1, test F1=0.4238):
  1. DISABLED manifold mixup — root cause of parent's instability. The
     mixup + focal loss + cross-attention combination on 4 tokens creates
     contradictory gradient signals on 1,500 samples (feedback-proven).
  2. 3-layer TransformerEncoder (was 2-layer) with dim_ff=384 — directly
     mirrors tree-best node3-1-1-1-1-1-2 (test F1=0.5049) that upgraded from
     2→3 layers and achieved +0.031 improvement.
  3. Deeper prediction head: 256→384→19920 (was 256→256→19920) — addresses
     the head capacity bottleneck identified in feedback: "256→256 head lacks
     sufficient capacity post-fusion; parent used 832→384 which was effective."
  4. Reduced weight_decay 0.08→0.10 — tree-best uses 0.10 consistently;
     0.08 was identified as too aggressive for this lineage in feedback.
  5. Reduced label_smoothing 0.10→0.05 — tree-best uses 0.05; 0.10 reduces
     ability to produce confident minority-class predictions.
  6. Extended training: max_epochs=120, early_stopping_patience=30 — node
     2-2-1-1-2-1 (test F1=0.4843) showed late-phase improvements at epoch 71
     under extended training with patience=55.
  7. Tighter ReduceLROnPlateau lr_patience=12 (was 8) — allows the model to
     benefit from LR reduction at the right time (tree-best breakthrough
     happened immediately after first LR reduction at epoch 35).
  8. Top-5 checkpoint averaging (was top-3) — node2-2-1-1-2-1 used top-5
     to achieve near-zero generalization gap (val-test=0.001).
  9. Gradient clip reduced to 0.5 (was 1.0) — more conservative clipping to
     prevent the epoch-0 instability seen in parent.

Architecture:
  AIDO.Cell-10M LoRA → mean-pool  → [B, 256]                        (token 0: global)
  AIDO.Cell-10M LoRA → pert-pos   → [B, 256]                        (token 1: pert-specific)
  Frozen STRING GNN  → PPI embed  → [B, 256]                        (token 2: PPI context)
  Symbol CNN         → char feat  → [B, 64] → project → [B, 256]   (token 3: naming context)
  Stack: [B, 4, 256]
  → 3-layer TransformerEncoder (nhead=8, dim_ff=384, pre-norm)
  → mean-pool → [B, 256]
  → LayerNorm(256) → Linear(256, 384) → GELU → Dropout(0.5) → Linear(384, 3×6640)
  → reshape [B, 3, 6640]
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
from typing import Any, Dict, List, Optional, Tuple
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
FUSION_TOKEN_DIM = 256  # Dimension of each token fed to TransformerEncoder

# Class weights: proven tree-best configuration (node3-1-1-1-1-1-2, F1=0.5049)
CLASS_WEIGHTS = torch.tensor([6.0, 1.0, 12.0], dtype=torch.float32)

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
        gamma: float = 1.5,
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


# ─── Cross-Attention Fusion Module ────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """
    Fuse 4 feature tokens via a multi-layer TransformerEncoder.

    Input:  [B, 4, FUSION_TOKEN_DIM]  (4 projected feature tokens)
    Output: [B, FUSION_TOKEN_DIM]     (mean-pooled across 4 tokens)

    Architecture aligned with tree-best node3-1-1-1-1-1-2 (test F1=0.5049):
    - 4 tokens: (global_pool, pert_emb, ppi_token, sym_token)
    - 3-layer TransformerEncoder with nhead=8, dim_ff=384
    - Pre-norm (norm_first=True) for better stability
    - Mean-pool over token dimension → [B, 256]
    """

    def __init__(
        self,
        d_model: int = FUSION_TOKEN_DIM,
        nhead: int = 8,
        n_layers: int = 3,
        dim_ff: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # [B, S, D] convention
            norm_first=True,    # Pre-norm for better training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, 4, d_model] float32
        Returns: [B, d_model] (mean-pooled fused representation)
        """
        out = self.encoder(tokens)      # [B, 4, d_model]
        return out.mean(dim=1)          # [B, d_model]


# ─── Prediction Head ──────────────────────────────────────────────────────────
class PredictionHead(nn.Module):
    """
    Deeper MLP head: [B, 256] → [B, N_CLASSES * N_GENES].

    Uses 256→384→19920 architecture (vs parent's 256→256→19920) to address
    the capacity bottleneck identified in feedback. The 384-dim intermediate
    layer provides more representational capacity to map the fused 256-dim
    representation to 19,920 output slots.
    """

    def __init__(
        self,
        in_dim: int = FUSION_TOKEN_DIM,
        hidden_dim: int = 384,
        out_dim: int = N_CLASSES * N_GENES,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
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
        # Resolve data_dir relative to cwd (symlink-aware when cd'd into a symlink).
        if Path(data_dir).is_absolute():
            self.data_dir = Path(data_dir)
        else:
            script_dir = Path(__file__).parent
            cwd = Path(os.getcwd())
            # Priority: script_based first, then cwd_based
            script_based = (script_dir / data_dir).resolve()
            if script_based.exists():
                self.data_dir = script_based
            else:
                self.data_dir = (cwd / data_dir).resolve()
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def _load_gnn_name_to_idx(self) -> Dict[str, int]:
        """Load STRING GNN node_names.json → {ensembl_id: node_index}."""
        node_names: List[str] = json.loads(
            (Path(STRING_GNN_PATH) / "node_names.json").read_text()
        )
        return {name: i for i, name in enumerate(node_names)}

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

        input_ids_tensor = torch.stack(all_input_ids)        # [N, 19264]
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
        n_fusion_layers: int = 3,
        fusion_ff_dim: int = 384,
        fusion_dropout: float = 0.1,
        head_hidden_dim: int = 384,
        head_dropout: float = 0.5,
        lr_backbone: float = 2e-4,
        lr_head: float = 6e-4,
        weight_decay: float = 0.10,
        gamma_focal: float = 1.5,
        label_smoothing: float = 0.05,
        lr_patience: int = 12,
        lr_factor: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialized in setup()
        self.backbone: Optional[nn.Module] = None
        self.sym_proj: Optional[nn.Linear] = None     # [64] → [256]
        self.symbol_encoder: Optional[SymbolCNN] = None
        self.fusion: Optional[CrossAttentionFusion] = None
        self.head: Optional[PredictionHead] = None
        self.criterion: Optional[FocalLoss] = None

        # Accumulators
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

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

        # ── 2. Symbol CNN + projection [64] → [256] ───────────────────────
        self.symbol_encoder = SymbolCNN(
            vocab_size=VOCAB_SIZE,
            embed_dim=32,
            out_dim=SYMBOL_OUT_DIM,
            max_len=MAX_SYMBOL_LEN,
        )
        self.sym_proj = nn.Linear(SYMBOL_OUT_DIM, FUSION_TOKEN_DIM)

        # ── 3. Frozen STRING GNN node embeddings ──────────────────────────
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

        # ── 4. Cross-attention fusion (3-layer, dim_ff=384) ───────────────
        self.fusion = CrossAttentionFusion(
            d_model=FUSION_TOKEN_DIM,
            nhead=8,
            n_layers=self.hparams.n_fusion_layers,
            dim_ff=self.hparams.fusion_ff_dim,
            dropout=self.hparams.fusion_dropout,
        )

        # ── 5. Prediction head (256→384→19920) ───────────────────────────
        self.head = PredictionHead(
            in_dim=FUSION_TOKEN_DIM,
            hidden_dim=self.hparams.head_hidden_dim,
            out_dim=N_CLASSES * N_GENES,
            dropout=self.hparams.head_dropout,
        )

        # ── 6. Focal Loss ─────────────────────────────────────────────────
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

    def _extract_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Extract and project all source features into 4 fusion tokens.
        Returns: tokens [B, 4, FUSION_TOKEN_DIM]

        Token layout:
          0 = AIDO global mean-pool (cell-state context)
          1 = AIDO per-gene positional embedding (pert-specific context)
          2 = STRING PPI network embedding (protein interaction context)
          3 = Symbol CNN embedding (gene naming/family context)
        """
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

        # Token 0: global mean-pool over all genes → [B, 256]
        mean_pool = hidden[:, :19264, :].mean(dim=1)   # [B, 256]
        global_token = mean_pool                       # [B, 256] (AIDO dim = FUSION_TOKEN_DIM)

        # Token 1: per-gene positional embedding for perturbed gene → [B, 256]
        # fallback to mean_pool for genes not found in AIDO vocabulary
        pert_emb = mean_pool.clone()
        valid_pos = pert_pos >= 0
        if valid_pos.any():
            pert_emb[valid_pos] = hidden[valid_pos, pert_pos[valid_pos], :]
        pert_token = pert_emb                          # [B, 256]

        # Token 2: frozen STRING GNN PPI embedding → [B, 256]
        ppi_token = torch.zeros(B, GNN_HIDDEN_DIM, device=self.device, dtype=torch.float32)
        valid_gnn = gnn_idx >= 0
        if valid_gnn.any():
            ppi_token[valid_gnn] = self.gnn_node_emb[gnn_idx[valid_gnn]]
        # GNN dim is already 256 = FUSION_TOKEN_DIM, no projection needed

        # Token 3: Symbol CNN → project [64] → [256]
        sym_feats = self.symbol_encoder(symbol_chars)  # [B, 64]
        sym_token = self.sym_proj(sym_feats)            # [B, 256]

        # Stack into sequence: [B, 4, 256]
        tokens = torch.stack([global_token, pert_token, ppi_token, sym_token], dim=1)
        return tokens

    # ── Forward pass ──────────────────────────────────────────────────────
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Standard forward (no mixup). Returns logits [B, 3, 6640]."""
        B = batch["input_ids"].shape[0]
        tokens = self._extract_features(batch)        # [B, 4, 256]
        fused = self.fusion(tokens)                   # [B, 256]
        logits = self.head(fused)                     # [B, 3 * N_GENES]
        return logits.view(B, N_CLASSES, N_GENES)     # [B, 3, 6640]

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

        # Compute per-rank F1, then average across ranks
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

        self.log("val_f1", f1, prog_bar=True, sync_dist=True)

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

            # Deduplicate by unique idx values. Since all ranks process the SAME
            # samples in the SAME order (use_distributed_sampler=False), the first
            # occurrence of each idx is the correct prediction.
            _, unique_pos = np.unique(idxs, return_index=True)
            preds = preds[unique_pos]
            sorted_idxs = idxs[unique_pos]

            # Sort by original idx order (restores 0,1,2,...,166 ordering)
            order = np.argsort(sorted_idxs)
            preds = preds[order]

            # n_expected is the full test set size (167).
            n_expected = len(self._test_pert_ids)
            n_rows = min(preds.shape[0], n_expected)
            if preds.shape[0] < n_expected:
                self.print(
                    f"WARNING: Collected {preds.shape[0]} < {n_expected} "
                    f"(limit_test_batches was active). Saving {n_rows} rows."
                )
            preds = preds[:n_rows]

            # Write test_predictions.tsv (same priority as main())
            script_dir = Path(__file__).parent
            cwd = Path(os.getcwd())
            script_based_run = (script_dir / "run").resolve()
            cwd_based_run = (cwd / "run").resolve()
            if script_based_run.exists():
                output_dir = script_based_run
            else:
                output_dir = cwd_based_run
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "test_predictions.tsv"

            rows = []
            for row_i in range(n_rows):
                rows.append({
                    "idx": self._test_pert_ids[row_i],
                    "input": self._test_symbols[row_i],
                    "prediction": json.dumps(preds[row_i].tolist()),
                })
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved ({n_rows} rows) → {out_path}")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    def configure_optimizers(self):
        backbone_params = [
            p for p in self.backbone.parameters() if p.requires_grad
        ]
        head_params = (
            list(self.sym_proj.parameters())
            + list(self.symbol_encoder.parameters())
            + list(self.fusion.parameters())
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


# ─── Top-K Checkpoint Averaging Callback ──────────────────────────────────────
class TopKCheckpointAveraging:
    """
    After training, average the top-k model checkpoints before test.
    Called manually after trainer.fit().
    """

    @staticmethod
    def load_and_average(
        model: DEGLightningModule,
        ckpt_dir: Path,
        top_k: int = 5,
        monitor: str = "val_f1",
        mode: str = "max",
    ) -> Optional[str]:
        """
        Find top-k checkpoints by monitored metric, average their trainable
        parameters, and save the averaged checkpoint.

        Returns the path to the averaged checkpoint (or None if fewer than top_k).
        """
        import re
        ckpt_files = list(ckpt_dir.glob("*.ckpt"))
        if not ckpt_files:
            print("[TopKAvg] No checkpoints found; skipping averaging.")
            return None

        # Parse metric values from filenames like "best-epoch=032-val_f1=0.4747.ckpt"
        scored = []
        for f in ckpt_files:
            name = f.stem
            m = re.search(r"val_f1=([0-9]+\.[0-9]+)", name)
            if m:
                val = float(m.group(1))
                scored.append((val, str(f)))

        if not scored:
            print("[TopKAvg] Cannot parse metric from checkpoint filenames; skipping averaging.")
            return None

        # Sort by metric
        scored.sort(key=lambda x: x[0], reverse=(mode == "max"))
        top_ckpts = scored[:min(top_k, len(scored))]
        print(f"[TopKAvg] Averaging {len(top_ckpts)} checkpoints:")
        for score, path in top_ckpts:
            print(f"  val_f1={score:.4f}  {path}")

        # Load and average state dicts
        avg_sd: Dict[str, torch.Tensor] = {}
        n = len(top_ckpts)
        for i, (score, path) in enumerate(top_ckpts):
            sd = torch.load(path, map_location="cpu", weights_only=False)
            # Lightning checkpoints wrap the model state under 'state_dict'
            model_sd = sd.get("state_dict", sd)
            if i == 0:
                for k, v in model_sd.items():
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        avg_sd[k] = v.float() / n
                    else:
                        avg_sd[k] = v
            else:
                for k, v in model_sd.items():
                    if k in avg_sd and isinstance(v, torch.Tensor) and v.is_floating_point():
                        avg_sd[k] = avg_sd[k] + v.float() / n

        # Save averaged checkpoint
        avg_path = str(ckpt_dir / "averaged_top5.ckpt")
        # Reuse the first checkpoint's metadata structure
        first_ckpt = torch.load(top_ckpts[0][1], map_location="cpu", weights_only=False)
        first_ckpt["state_dict"] = avg_sd
        torch.save(first_ckpt, avg_path)
        print(f"[TopKAvg] Averaged checkpoint saved → {avg_path}")
        return avg_path


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node 1-2 (Stabilized): Cross-Attention Fusion DEG Predictor (no mixup)"
    )
    # Data / output
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-workers", type=int, default=4)
    # Batch sizes
    parser.add_argument("--micro-batch-size", type=int, default=4,
                        help="Batch size per GPU.")
    parser.add_argument("--global-batch-size", type=int, default=32,
                        help="Effective batch size. Must be multiple of micro_batch_size * 8.")
    # Training
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    # LoRA
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    # Fusion transformer
    parser.add_argument("--n-fusion-layers", type=int, default=3,
                        help="Number of TransformerEncoder layers in fusion module.")
    parser.add_argument("--fusion-ff-dim", type=int, default=384,
                        help="Feedforward dim in TransformerEncoder layers.")
    parser.add_argument("--fusion-dropout", type=float, default=0.1)
    # Head
    parser.add_argument("--head-hidden-dim", type=int, default=384)
    parser.add_argument("--head-dropout", type=float, default=0.5)
    # Optimizer
    parser.add_argument("--lr-backbone", type=float, default=2e-4)
    parser.add_argument("--lr-head", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.10)
    # Scheduler + Early stopping
    parser.add_argument("--lr-patience", type=int, default=12)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    # Loss
    parser.add_argument("--gamma-focal", type=float, default=1.5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    # Top-k checkpoint averaging (5 for better variance reduction)
    parser.add_argument("--top-k-avg", type=int, default=5,
                        help="Average top-k checkpoints at test time. Set 0 to use best only.")
    # Debug
    parser.add_argument("--debug-max-step", type=int, default=None,
                        help="Limit global training steps for quick debugging.")
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Run 1 batch for train/val/test (smoke test).")
    args = parser.parse_args()

    # ── Output dir ────────────────────────────────────────────────────────
    # Prefer script-based resolution for run/ (stays in node directory).
    # Fall back to cwd-based if script-based doesn't exist.
    script_dir = Path(__file__).parent
    cwd = Path(os.getcwd())
    script_based_run = (script_dir / "run").resolve()
    cwd_based_run = (cwd / "run").resolve()
    # Priority: script_based_run if it exists, else cwd_based_run
    if script_based_run.exists():
        output_dir = script_based_run
    else:
        output_dir = cwd_based_run
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
        save_top_k=5,     # Save top-5 for checkpoint averaging
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=10)
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
        gradient_clip_val=0.5,       # Reduced from 1.0 to prevent epoch-0 instability
        use_distributed_sampler=False,  # All ranks process ALL test samples in same order
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
        n_fusion_layers=args.n_fusion_layers,
        fusion_ff_dim=args.fusion_ff_dim,
        fusion_dropout=args.fusion_dropout,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
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

    # ── Test with checkpoint averaging ────────────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        ckpt_path = "best"
        # Attempt top-k checkpoint averaging if requested
        if args.top_k_avg > 0 and not args.fast_dev_run:
            avg_path = TopKCheckpointAveraging.load_and_average(
                model,
                ckpt_dir=output_dir / "checkpoints",
                top_k=args.top_k_avg,
                monitor="val_f1",
                mode="max",
            )
            if avg_path is not None:
                ckpt_path = avg_path
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

    if test_results:
        print(f"Test results: {test_results}")

    # ── Save test score ───────────────────────────────────────────────────────
    if trainer.is_global_zero:
        score_path = output_dir / "test_score.txt"
        best_val_f1 = (
            float(checkpoint_callback.best_model_score)
            if checkpoint_callback.best_model_score is not None
            else None
        )
        score_path.write_text(
            f"test_results: {test_results}\n"
            f"val_f1_best: {best_val_f1}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
