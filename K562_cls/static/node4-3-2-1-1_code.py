"""Node 4-3-2-1-2: scFoundation(6L FT) + STRING_GNN(frozen) + GatedFusion
+ Two-stage GenePriorBias(scale=0.5, warmup=50) + Gene-Freq Loss (factor=2.0)
+ Stochastic Weight Averaging (SWA, start=120) + Extended Training.

Key changes from parent node4-3-2-1 (F1=0.4601):
1. REVERT min_lr_ratio: 0.02 → 0.05
   The parent's 0.02 caused a catastrophic epoch-37 val F1 collapse (0.445→0.346)
   and persistent oscillations. The proven value from node4-3-2 (F1=0.4650) and
   node4-2-1-2 (F1=0.4893, scFoundation lineage record) is 0.05.
2. REVERT gene_diversity_factor: 4.0 → 2.0
   Parent feedback confirmed factor=4.0 destabilized the fusion loss landscape.
   Factor=2.0 was used in node4-2-1-2 (F1=0.4893) — the top scFoundation node.
3. REMOVE 3-checkpoint ensemble → Replace with SWA (Stochastic Weight Averaging)
   The parent's 3-ckpt ensemble added zero diversity (all F1=0.4599–0.4601).
   SWA averages weights over late training epochs, providing a fundamentally
   different form of model averaging that has shown +0.003–0.005 F1 gains
   in the SWA literature and in the scFoundation lineage (node4-3, SWA with
   68 checkpoints → node4-3 baseline F1=0.4585). The key insight: SWA with
   a reasonable number of epochs (not 68 like node4-3's extreme case) should
   generalize without overfitting.
4. EXTEND training: max_epochs=400, patience=35, min_epochs=100
   The parent stopped at epoch 80 (only 10 epochs past best). Extended training
   with higher patience allows the model to reach the node4-2-1-2 trajectory
   (which peaked at epoch 167) after GenePriorBias activation at epoch 50.
5. Use save_top_k=1 (single best checkpoint) for test inference.
   No ensemble — ensemble showed zero value in the parent due to identical
   checkpoint F1 values.

Architecture:
    scFoundation(6L FT) → mean_pool → [B,768]
    STRING_GNN(frozen) → [B,256]
    GatedFusion(768+256→512) → Mixup → 2-layer Head
    + Two-stage GenePriorBias[3,6640] (frozen until epoch 50, then scale=0.5)
    → [B,3,6640] logits → softmax probs

Training:
    WarmupCosine(warmup=10, min_lr_ratio=0.05, max_epochs=400)
    SWA start_epoch=120, swa_lr=5e-5, update every 3 epochs
    EarlyStopping(patience=35, min_delta=1e-4, min_epochs=100)
    Mixup(alpha=0.2), gene_diversity_factor=2.0, batch_size=16
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES    = 6640
N_CLASSES  = 3
SCF_HIDDEN = 768    # scFoundation hidden size
GNN_HIDDEN = 256    # STRING_GNN hidden size
FUSION_DIM = 512    # output dimension of gated fusion
HEAD_HIDDEN = 256   # intermediate dimension of 2-layer head

SCF_MODEL_DIR = "/home/Models/scFoundation"
GNN_MODEL_DIR = "/home/Models/STRING_GNN"

# Class frequency: down(-1→0), neutral(0→1), up(+1→2)
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Inverse-sqrt-frequency class weights for neutral class suppression."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_gene_priors() -> torch.Tensor:
    """Compute per-gene log-class-frequency priors from training data.

    Returns: [3, N_GENES] float32 tensor of log-class probabilities per gene.
    Used to initialize GenePriorBias.
    """
    train_df = pd.read_csv(TRAIN_TSV, sep="\t")
    counts = np.zeros((N_CLASSES, N_GENES), dtype=np.float64)
    for label_str in train_df["label"]:
        labels_arr = np.array(json.loads(label_str)) + 1  # {-1,0,1} → {0,1,2}
        for c in range(N_CLASSES):
            counts[c] += (labels_arr == c).astype(np.float64)
    # Laplace smoothing to avoid log(0)
    counts += 1.0
    total = counts.sum(axis=0, keepdims=True)
    log_priors = np.log(counts / total)
    return torch.tensor(log_priors, dtype=torch.float32)


def compute_gene_frequency_weights(n_top: int = 2000, base_weight: float = 1.0,
                                    boost_weight: float = 2.0) -> torch.Tensor:
    """Compute per-gene loss weights based on DEG frequency.

    Top-n_top most DEG-variable genes get boost_weight; rest get base_weight.
    factor=2.0 is the proven stable setting from node4-2-1-2 (F1=0.4893).
    The parent's factor=4.0 caused loss landscape instability in the fusion context.

    Returns: [N_GENES] float32 tensor.
    """
    train_df = pd.read_csv(TRAIN_TSV, sep="\t")
    deg_count = np.zeros(N_GENES, dtype=np.float64)
    for label_str in train_df["label"]:
        labels_arr = np.array(json.loads(label_str))  # {-1,0,1}
        deg_count += (labels_arr != 0).astype(np.float64)  # count non-neutral

    # Top n_top most variable genes get boosted weight
    gene_weights = np.full(N_GENES, base_weight, dtype=np.float32)
    top_indices = np.argsort(deg_count)[-n_top:]
    gene_weights[top_indices] = boost_weight
    return torch.tensor(gene_weights, dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute per-gene macro-averaged F1, matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] softmax probabilities
        targets: [N, G]   integer labels in {0,1,2}
    Returns:
        scalar F1 averaged over all G genes
    """
    y_hat       = preds.argmax(dim=1)          # [N, G]
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
        f1_c = torch.where(
            prec + rec > 0,
            2 * prec * rec / (prec + rec + 1e-8),
            torch.zeros_like(prec),
        )
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# Cosine LR Schedule with Linear Warmup
# ---------------------------------------------------------------------------
def get_warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """LambdaLR schedule: linear warmup then cosine decay to min_lr_ratio * base_lr.

    min_lr_ratio=0.05 is the proven value from node4-3-2 (0.4650) and node4-2-1-2
    (0.4893). The parent used 0.02 which caused catastrophic optimization instability.
    """
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Gated Fusion Module
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Learnable gated fusion of scFoundation and STRING_GNN embeddings.

    output = LN(gate_scf * proj_scf(scf) + gate_gnn * proj_gnn(gnn)) + Dropout
    Gates are computed from the concatenation of both inputs.
    """

    def __init__(
        self,
        d_scf: int = SCF_HIDDEN,
        d_gnn: int = GNN_HIDDEN,
        d_out: int = FUSION_DIM,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        d_in = d_scf + d_gnn
        self.proj_scf   = nn.Linear(d_scf, d_out)
        self.proj_gnn   = nn.Linear(d_gnn, d_out)
        self.gate_scf   = nn.Linear(d_in,  d_out)
        self.gate_gnn   = nn.Linear(d_in,  d_out)
        self.layer_norm = nn.LayerNorm(d_out)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, scf_emb: torch.Tensor, gnn_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([scf_emb, gnn_emb], dim=-1)
        gate_s   = torch.sigmoid(self.gate_scf(combined))
        gate_g   = torch.sigmoid(self.gate_gnn(combined))
        fused    = gate_s * self.proj_scf(scf_emb) + gate_g * self.proj_gnn(gnn_emb)
        return self.dropout(self.layer_norm(fused))       # [B, FUSION_DIM]


# ---------------------------------------------------------------------------
# Dataset & DataModule
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


def make_collate_scf(tokenizer):
    """Collate function that tokenizes for scFoundation."""

    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]

        # scFoundation: only the perturbed gene has expression=1.0; all others → 0.0
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")

        out: Dict[str, Any] = {
            "sample_idx":     torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":        pert_ids,
            "symbol":         symbols,
            "input_ids":      tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
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
            AutoTokenizer.from_pretrained(SCF_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(SCF_MODEL_DIR, trust_remote_code=True)

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")
        self.train_ds = DEGDataset(train_df)
        self.val_ds   = DEGDataset(val_df)
        self.test_ds  = DEGDataset(test_df)

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size   = self.batch_size,
            shuffle      = shuffle,
            num_workers  = self.num_workers,
            pin_memory   = True,
            collate_fn   = make_collate_scf(self.tokenizer),
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# SWA Helper: In-place SWA averaging on the model's trainable parameters
# ---------------------------------------------------------------------------
class SWAHelper:
    """Stochastic Weight Averaging (SWA) helper.

    Maintains running average of model weights. At test time, the averaged
    weights provide a smoother, often more generalizable model than any
    individual checkpoint.

    Usage:
        swa = SWAHelper(model)
        for epoch in training:
            if epoch >= swa_start_epoch and epoch % swa_freq == 0:
                swa.update(model)
        swa.apply(model)  # Set model to averaged weights for testing
        swa.restore(model)  # Restore original weights if needed
    """

    def __init__(self) -> None:
        self._swa_weights: Optional[Dict[str, torch.Tensor]] = None
        self._n_models: int = 0

    def update(self, model: nn.Module) -> None:
        """Update SWA running average with current model weights."""
        state = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                state[name] = param.data.detach().clone().cpu()
        for name, buf in model.named_buffers():
            state[name] = buf.data.detach().clone().cpu()

        if self._swa_weights is None:
            self._swa_weights = state
            self._n_models = 1
        else:
            n = self._n_models
            for name in self._swa_weights:
                if name in state:
                    self._swa_weights[name] = (
                        self._swa_weights[name] * n / (n + 1) +
                        state[name] / (n + 1)
                    )
            self._n_models = n + 1

    def apply(self, model: nn.Module) -> None:
        """Apply SWA averaged weights to model."""
        if self._swa_weights is None or self._n_models == 0:
            return
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self._swa_weights:
                    param.data.copy_(
                        self._swa_weights[name].to(param.device).to(param.dtype)
                    )
            for name, buf in model.named_buffers():
                if name in self._swa_weights:
                    buf.data.copy_(
                        self._swa_weights[name].to(buf.device).to(buf.dtype)
                    )

    def is_active(self) -> bool:
        return self._n_models > 0

    @property
    def n_models(self) -> int:
        return self._n_models


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FusionDEGModel(pl.LightningModule):
    """scFoundation (top-6L fine-tuned) + STRING_GNN (frozen cached)
    + GatedFusion + Two-stage GenePriorBias + Gene-frequency-weighted loss
    + Stochastic Weight Averaging (SWA).

    Key differences from parent node4-3-2-1 (F1=0.4601):
    - min_lr_ratio reverted to 0.05 (parent's 0.02 caused epoch-37 catastrophic collapse)
    - gene_diversity_factor reverted to 2.0 (4.0 destabilized fusion loss landscape)
    - SWA added: averages weights from epoch 120 onwards, every 3 epochs
    - Extended training: max_epochs=400, patience=35, min_epochs=100
    - Single best checkpoint for test (no ensemble — showed zero value in parent)
    """

    def __init__(
        self,
        scf_finetune_layers: int  = 6,
        head_dropout: float       = 0.5,
        head_hidden: int          = HEAD_HIDDEN,
        fusion_dropout: float     = 0.3,
        lr: float                 = 2e-4,
        weight_decay: float       = 3e-2,
        label_smoothing: float    = 0.1,
        warmup_epochs: int        = 10,
        min_lr_ratio: float       = 0.05,
        max_epochs: int           = 400,
        mixup_alpha: float        = 0.2,
        bias_warmup_epochs: int   = 50,
        prior_scale: float        = 0.5,
        gene_diversity_factor: float = 2.0,
        swa_start_epoch: int      = 120,
        swa_freq: int             = 3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        gnn_dir = Path(GNN_MODEL_DIR)

        # ------------------------------------------------------------------
        # 1. scFoundation backbone (partial fine-tune: top-6 of 12 layers)
        # ------------------------------------------------------------------
        self.scf = AutoModel.from_pretrained(
            SCF_MODEL_DIR,
            trust_remote_code=True,
            _use_flash_attention_2=True,
        )
        self.scf = self.scf.to(torch.bfloat16)
        self.scf.config.use_cache = False
        self.scf.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Freeze all scF params, then unfreeze top-k transformer layers
        for param in self.scf.parameters():
            param.requires_grad = False

        n_scf_layers = len(self.scf.encoder.transformer_encoder)
        for i in range(n_scf_layers - hp.scf_finetune_layers, n_scf_layers):
            for param in self.scf.encoder.transformer_encoder[i].parameters():
                param.requires_grad = True
        # Also unfreeze the final LayerNorm
        for param in self.scf.encoder.norm.parameters():
            param.requires_grad = True

        # Cast unfrozen scF params to float32 for stable optimization
        for name, param in self.scf.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        scf_train = sum(p.numel() for p in self.scf.parameters() if p.requires_grad)
        scf_total = sum(p.numel() for p in self.scf.parameters())
        print(f"[Node4-3-2-1-2] scFoundation: {scf_train:,}/{scf_total:,} trainable")

        # ------------------------------------------------------------------
        # 2. STRING_GNN: precompute embeddings once, store as frozen buffer
        # No NeighborhoodAttentionModule — proven configuration from node4-2-1-2
        # (F1=0.4893, scFoundation lineage record): raw cached embeddings directly
        # to GatedFusion without the neighborhood attention overhead.
        # ------------------------------------------------------------------
        gnn = AutoModel.from_pretrained(str(gnn_dir), trust_remote_code=True)
        gnn = gnn.to(torch.float32)

        graph_data = torch.load(gnn_dir / "graph_data.pt", map_location="cpu")
        edge_index  = graph_data["edge_index"].long()
        edge_weight = graph_data.get("edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.float()

        with torch.no_grad():
            gnn_out = gnn(edge_index=edge_index, edge_weight=edge_weight)
            gnn_embs = gnn_out.last_hidden_state.float()   # [N_nodes, 256]

        self.register_buffer("gnn_embs_cached", gnn_embs)  # [18870, 256]

        node_names = json.loads((gnn_dir / "node_names.json").read_text())
        self._ensembl_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(node_names)
        }

        del gnn  # Free GNN model memory
        print(f"[Node4-3-2-1-2] STRING_GNN: frozen, cached {gnn_embs.shape} embeddings")

        # ------------------------------------------------------------------
        # 3. Gated Fusion
        # ------------------------------------------------------------------
        self.fusion = GatedFusion(
            d_scf    = SCF_HIDDEN,
            d_gnn    = GNN_HIDDEN,
            d_out    = FUSION_DIM,
            dropout  = hp.fusion_dropout,
        )

        # ------------------------------------------------------------------
        # 4. Two-layer Classification Head
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Dropout(hp.head_dropout),
            nn.Linear(FUSION_DIM, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout * 0.5),      # half dropout for second layer
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )

        # ------------------------------------------------------------------
        # 5. Two-stage GenePriorBias (from node4-2-1-2, zero val-test gap)
        # Frozen for first bias_warmup_epochs, then enabled at prior_scale=0.5
        # ------------------------------------------------------------------
        priors = compute_gene_priors()                        # [3, N_GENES]
        self.gene_prior_bias = nn.Parameter(priors)           # learnable [3, N_GENES]
        print(f"[Node4-3-2-1-2] GenePriorBias initialized, scale={hp.prior_scale}, "
              f"warmup={hp.bias_warmup_epochs} epochs")

        # ------------------------------------------------------------------
        # 6. Loss: weighted cross-entropy + gene-frequency-weighted loss
        # gene_diversity_factor=2.0 (reverted from parent's 4.0)
        # 4.0 caused optimization instability — 2.0 is the proven stable value
        # from node4-2-1-2 (F1=0.4893).
        # ------------------------------------------------------------------
        self.register_buffer("class_weights", get_class_weights())
        self._label_smoothing = hp.label_smoothing

        # Gene-frequency-based per-gene weights (factor=2.0, reverted from 4.0)
        gene_freq_weights = compute_gene_frequency_weights(
            n_top=2000,
            base_weight=1.0,
            boost_weight=hp.gene_diversity_factor,
        )
        self.register_buffer("gene_freq_weights", gene_freq_weights)  # [N_GENES]

        # ------------------------------------------------------------------
        # 7. SWA helper
        # ------------------------------------------------------------------
        self._swa = SWAHelper()
        self._swa_apply_at_test_start = False  # Set to True before test to apply SWA

        # Prediction buffers
        self._val_preds:      List[torch.Tensor] = []
        self._val_tgts:       List[torch.Tensor] = []
        self._val_idx:        List[torch.Tensor] = []
        self._test_preds:     List[torch.Tensor] = []
        self._test_tgts:      List[torch.Tensor] = []
        self._test_pert_ids:  List[str] = []
        self._test_symbols:   List[str] = []

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _get_gnn_emb(self, pert_ids: List[str], device: torch.device) -> torch.Tensor:
        """Look up cached GNN embeddings for a batch of Ensembl IDs.

        Returns raw cached embeddings directly to GatedFusion.
        Proven configuration from node4-2-1-2 (F1=0.4893, scFoundation lineage record).
        """
        indices = [self._ensembl_to_idx.get(pid, 0) for pid in pert_ids]
        idx_t   = torch.tensor(indices, dtype=torch.long, device=device)
        return self.gnn_embs_cached[idx_t]             # [B, 256]

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids: List[str],
    ) -> torch.Tensor:
        B      = input_ids.shape[0]
        device = input_ids.device

        # 1. scFoundation embedding: [B, nnz+2, 768] → mean pool → [B, 768]
        scf_out = self.scf(input_ids=input_ids, attention_mask=attention_mask)
        scf_emb = scf_out.last_hidden_state.float().mean(dim=1)   # [B, 768]

        # 2. STRING_GNN embedding (raw, no neighborhood attention): [B, 256]
        gnn_emb = self._get_gnn_emb(pert_ids, device)

        # 3. Gated fusion: [B, 512]
        fused = self.fusion(scf_emb, gnn_emb)

        # 4. Classification head: [B, N_CLASSES*N_GENES] → [B, 3, G]
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)

        # 5. Two-stage GenePriorBias: applied at prior_scale after warmup
        # Scale factor = 0.5 so prior doesn't dominate early (node4-2-1-2 proven)
        prior_scale = (
            self.hparams.prior_scale
            if self.current_epoch >= self.hparams.bias_warmup_epochs
            else 0.0
        )
        logits = logits + prior_scale * self.gene_prior_bias.to(logits.dtype)    # [B, 3, G]

        return logits

    # -----------------------------------------------------------------------
    # Loss: gene-frequency-weighted cross-entropy (factor=2.0 for top-2000 DEG genes)
    # -----------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy with per-gene frequency weighting.

        Per-gene weights up-weight DEG-variable genes (top-2000 × 2.0).
        factor=2.0 is the proven stable value from node4-2-1-2 (F1=0.4893).
        """
        B, C, G = logits.shape
        # Standard weighted CE per gene-sample pair
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
        targets_flat = targets.reshape(-1)                      # [B*G]

        # Per-sample-gene weights: [B*G] = broadcast gene_freq_weights over batch
        sample_gene_weights = self.gene_freq_weights.unsqueeze(0).expand(B, -1).reshape(-1)
        # Normalize within this batch to maintain mean=1.0
        sample_gene_weights = sample_gene_weights / sample_gene_weights.mean()

        # Compute per-element CE loss (unreduced)
        per_element_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            weight          = self.class_weights.to(logits.device),
            label_smoothing = self._label_smoothing,
            reduction       = "none",
        )

        # Apply gene-frequency weights
        weighted_loss = (per_element_loss * sample_gene_weights).mean()
        return weighted_loss

    # -----------------------------------------------------------------------
    # Mixup augmentation (applied in embedding space)
    # -----------------------------------------------------------------------
    def _mixup_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids: List[str],
        targets: torch.Tensor,
        alpha: float = 0.2,
    ):
        """Mixup in fused embedding space during training."""
        B      = input_ids.shape[0]
        device = input_ids.device

        # Standard forward up to fusion
        scf_out = self.scf(input_ids=input_ids, attention_mask=attention_mask)
        scf_emb = scf_out.last_hidden_state.float().mean(dim=1)
        gnn_emb = self._get_gnn_emb(pert_ids, device)
        fused   = self.fusion(scf_emb, gnn_emb)              # [B, 512]

        # Sample lambda from Beta distribution
        lam  = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        lam  = max(lam, 1 - lam)                              # ensure lam >= 0.5
        perm = torch.randperm(B, device=device)

        fused_mixed = lam * fused + (1 - lam) * fused[perm]  # [B, 512]

        # Head + bias (use same two-stage scaling)
        logits = self.head(fused_mixed).view(B, N_CLASSES, N_GENES)
        prior_scale = (
            self.hparams.prior_scale
            if self.current_epoch >= self.hparams.bias_warmup_epochs
            else 0.0
        )
        logits = logits + prior_scale * self.gene_prior_bias.to(logits.dtype)

        # Mixed loss
        loss_a = self._loss(logits, targets)
        loss_b = self._loss(logits, targets[perm])
        return lam * loss_a + (1 - lam) * loss_b

    # -----------------------------------------------------------------------
    # Steps
    # -----------------------------------------------------------------------
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        hp = self.hparams
        if hp.mixup_alpha > 0 and "labels" in batch:
            loss = self._mixup_forward(
                batch["input_ids"], batch["attention_mask"],
                batch["pert_id"], batch["labels"],
                alpha=hp.mixup_alpha,
            )
        else:
            logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
            loss   = self._loss(logits, batch["labels"])

        self.log("train/loss", loss, prog_bar=True, sync_dist=True,
                 on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self) -> None:
        """Update SWA running average at specified epochs."""
        hp = self.hparams
        epoch = self.current_epoch
        if epoch >= hp.swa_start_epoch and (epoch - hp.swa_start_epoch) % hp.swa_freq == 0:
            self._swa.update(self)
            self.print(
                f"[Node4-3-2-1-2] SWA update at epoch {epoch} "
                f"(total averaged: {self._swa.n_models})"
            )

    def on_after_backward(self) -> None:
        """Zero GenePriorBias gradient during warmup epochs.

        The forward pass already uses prior_scale=0.0 during warmup, but we also
        zero the gradient to ensure no accidental updates during warmup.
        """
        if self.current_epoch < self.hparams.bias_warmup_epochs:
            if self.gene_prior_bias.grad is not None:
                self.gene_prior_bias.grad.zero_()

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
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
        s_idx  = all_idx[order]
        s_pred = all_preds[order]
        s_tgt  = all_tgts[order]
        mask   = torch.cat([
            torch.tensor([True], device=s_idx.device),
            s_idx[1:] != s_idx[:-1],
        ])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        # NOTE: 'val_f1' (underscore) used to avoid '/' in checkpoint filenames.
        # PyTorch interprets '/' as a path separator in checkpoint filenames.
        # The monitor in ModelCheckpoint and EarlyStopping must also use 'val_f1'.
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)

    def on_test_start(self) -> None:
        """Apply SWA weights at the start of testing if flagged."""
        if getattr(self, "_swa_apply_at_test_start", False):
            if self._swa.is_active() and self._swa.n_models >= 3:
                self._swa.apply(self)
                self.print(
                    f"[Node4-3-2-1-2] Applied SWA weights from "
                    f"{self._swa.n_models} averaged models."
                )
            self._swa_apply_at_test_start = False

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)
            self._test_tgts.append(batch["labels"].detach())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)
        all_preds   = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)

        # Gather pert_ids and symbols from ALL ranks (collective must run on all)
        is_distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if is_distributed:
            all_pert_ids: List[List[str]] = [None] * self.trainer.world_size
            all_symbols:  List[List[str]] = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(all_pert_ids, self._test_pert_ids)
            torch.distributed.all_gather_object(all_symbols,  self._test_symbols)
            flat_pids = [p for rank_pids in all_pert_ids for p in rank_pids]
            flat_syms = [s for rank_syms in all_symbols  for s in rank_syms]
        else:
            flat_pids = list(self._test_pert_ids)
            flat_syms = list(self._test_symbols)

        # Compute test F1 (all ranks compute from gathered tensors, same result)
        if self._test_tgts:
            local_tgts = torch.cat(self._test_tgts, 0)
            all_tgts   = self.all_gather(local_tgts).view(-1, N_GENES)
            # Sort by pert_id hash and dedupe (same as validation pattern)
            pid_hashes = torch.tensor([hash(p) % (2**63) for p in flat_pids])
            pid_order  = torch.argsort(pid_hashes)
            s_idx  = pid_hashes[pid_order].to(all_preds.device)
            s_pred = all_preds[pid_order]
            s_tgt  = all_tgts[pid_order]
            mask   = torch.cat([
                torch.tensor([True], device=s_pred.device),
                s_idx[1:] != s_idx[:-1],
            ])
            test_f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
            self.log("test/f1", test_f1, on_step=False, on_epoch=True, sync_dist=True)
            print(f"[Node4-3-2-1-2] Test F1: {test_f1:.4f}")

        # Save predictions only on global zero to avoid duplicate files
        if self.trainer.is_global_zero:
            n = all_preds.shape[0]
            rows = []
            seen_pids = set()
            for i in range(n):
                pid = flat_pids[i]
                if pid in seen_pids:
                    continue
                seen_pids.add(pid)
                rows.append({
                    "idx":        pid,
                    "input":      flat_syms[i],
                    "prediction": json.dumps(
                        all_preds[i].float().cpu().numpy().tolist()
                    ),
                })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(
                out_dir / "test_predictions.tsv", sep="\t", index=False
            )
            print(f"[Node4-3-2-1-2] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_tgts.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    # -----------------------------------------------------------------------
    # Checkpoint: save only trainable params + buffers
    # -----------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
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
        total    = sum(p.numel() for p in self.parameters())
        trained  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Checkpoint: {trained:,}/{total:,} params "
            f"({100 * trained / total:.2f}%) + buffers"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        full_keys = set(super().state_dict().keys())
        trainable_keys = {
            name for name, p in self.named_parameters() if p.requires_grad
        }
        buffer_keys = {
            name for name, _ in self.named_buffers() if name in full_keys
        }
        expected = trainable_keys | buffer_keys
        missing    = [k for k in expected    if k not in state_dict]
        unexpected = [k for k in state_dict  if k not in expected]
        if missing[:3]:
            self.print(f"Warning: Missing checkpoint keys (first 3): {missing[:3]}")
        if unexpected[:3]:
            self.print(f"Warning: Unexpected keys (first 3): {unexpected[:3]}")
        return super().load_state_dict(state_dict, strict=False)

    # -----------------------------------------------------------------------
    # Optimizer: decoupled weight decay (skip LN/bias/GenePriorBias)
    # -----------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams

        # Separate parameter groups: no weight decay for normalization/bias params
        no_decay_names = set()
        for name, _ in self.named_parameters():
            # LayerNorm params, bias terms, and GenePriorBias
            if (name.endswith(".bias") or
                    "layer_norm" in name.lower() or
                    "layernorm" in name.lower() or
                    "norm" in name.lower() or
                    "gene_prior_bias" in name):
                no_decay_names.add(name)

        decay_params   = [p for n, p in self.named_parameters()
                          if p.requires_grad and n not in no_decay_names]
        no_decay_params = [p for n, p in self.named_parameters()
                           if p.requires_grad and n in no_decay_names]

        param_groups = [
            {"params": decay_params,    "weight_decay": hp.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        # Filter out empty groups to avoid AdamW warnings
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        opt = torch.optim.AdamW(param_groups, lr=hp.lr)
        sch = get_warmup_cosine_schedule(
            optimizer      = opt,
            warmup_epochs  = hp.warmup_epochs,
            total_epochs   = hp.max_epochs,
            min_lr_ratio   = hp.min_lr_ratio,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval":  "epoch",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node4-3-2-1-2: scFoundation + STRING_GNN(frozen) "
                    "+ GatedFusion + Two-stage GenePriorBias + Gene-freq loss(2.0) "
                    "+ SWA(start=120) + Extended Training"
    )
    parser.add_argument("--micro-batch-size",       type=int,   default=16)
    parser.add_argument("--global-batch-size",      type=int,   default=128)
    parser.add_argument("--max-epochs",             type=int,   default=400)
    parser.add_argument("--lr",                     type=float, default=2e-4)
    parser.add_argument("--weight-decay",           type=float, default=3e-2)
    parser.add_argument("--scf-finetune-layers",    type=int,   default=6,
                        dest="scf_finetune_layers")
    parser.add_argument("--head-dropout",           type=float, default=0.5)
    parser.add_argument("--fusion-dropout",         type=float, default=0.3)
    parser.add_argument("--label-smoothing",        type=float, default=0.1)
    parser.add_argument("--warmup-epochs",          type=int,   default=10)
    parser.add_argument("--min-lr-ratio",           type=float, default=0.05,
                        dest="min_lr_ratio")
    parser.add_argument("--mixup-alpha",            type=float, default=0.2,
                        dest="mixup_alpha")
    parser.add_argument("--bias-warmup-epochs",     type=int,   default=50,
                        dest="bias_warmup_epochs")
    parser.add_argument("--prior-scale",            type=float, default=0.5,
                        dest="prior_scale")
    parser.add_argument("--gene-diversity-factor",  type=float, default=2.0,
                        dest="gene_diversity_factor")
    parser.add_argument("--swa-start-epoch",        type=int,   default=120,
                        dest="swa_start_epoch")
    parser.add_argument("--swa-freq",               type=int,   default=3,
                        dest="swa_freq")
    parser.add_argument("--patience",               type=int,   default=35)
    parser.add_argument("--min-epochs",             type=int,   default=100,
                        dest="min_epochs")
    parser.add_argument("--num-workers",            type=int,   default=4)
    parser.add_argument("--debug-max-step",         type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--debug_max_step",         type=int,   default=None,
                        dest="debug_max_step",
                        help="Alias for --debug-max-step (underscore variant)")
    parser.add_argument("--fast-dev-run",           action="store_true",
                        dest="fast_dev_run")
    parser.add_argument("--fast_dev_run",           action="store_true",
                        dest="fast_dev_run",
                        help="Alias for --fast-dev-run (underscore variant)")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = lim_val = args.debug_max_step
        lim_test  = 1.0   # Always process ALL test samples even in debug mode
        max_steps = args.debug_max_step
    else:
        lim_train = lim_val = lim_test = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)

    model_kwargs = dict(
        scf_finetune_layers    = args.scf_finetune_layers,
        head_dropout           = args.head_dropout,
        head_hidden            = HEAD_HIDDEN,
        fusion_dropout         = args.fusion_dropout,
        lr                     = args.lr,
        weight_decay           = args.weight_decay,
        label_smoothing        = args.label_smoothing,
        warmup_epochs          = args.warmup_epochs,
        min_lr_ratio           = args.min_lr_ratio,
        max_epochs             = args.max_epochs,
        mixup_alpha            = args.mixup_alpha,
        bias_warmup_epochs     = args.bias_warmup_epochs,
        prior_scale            = args.prior_scale,
        gene_diversity_factor  = args.gene_diversity_factor,
        swa_start_epoch        = args.swa_start_epoch,
        swa_freq               = args.swa_freq,
    )
    model = FusionDEGModel(**model_kwargs)

    # CRITICAL: use `val_f1` (underscore) to avoid '/' causing torch.save failures
    # when '/' appears in checkpoint filenames (PyTorch interprets it as a path separator).
    # The metric is logged as 'val_f1' to match this filename convention.
    # save_top_k=1: single best checkpoint — ensemble showed zero value in parent.
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val_f1:.4f}",
        monitor    = "val_f1",
        mode       = "max",
        save_top_k = 1,      # Single best: ensemble added no diversity in parent
    )
    es_cb = EarlyStopping(
        monitor   = "val_f1",
        mode      = "max",
        patience  = args.patience,
        min_delta = 1e-4,    # Same as node4-3-2 proven value
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    strategy = (
        DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=600))
        if n_gpus > 1 else "auto"
    )

    # min_epochs=100 guards against early stopping before GenePriorBias warmup completes
    # and before the SWA region is reached (SWA starts at epoch 120)
    trainer = pl.Trainer(
        accelerator             = "gpu",
        devices                 = n_gpus,
        num_nodes               = 1,
        strategy                = strategy,
        precision               = "bf16-mixed",
        max_epochs              = args.max_epochs,
        min_epochs              = args.min_epochs,
        max_steps               = max_steps,
        accumulate_grad_batches = accum,
        limit_train_batches     = lim_train,
        limit_val_batches       = lim_val,
        limit_test_batches      = lim_test,
        val_check_interval      = 1.0,
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

    # -----------------------------------------------------------------------
    # Test: Apply SWA weights if available, then run on best checkpoint
    # SWA approach: load best ckpt, then overwrite weights with SWA average.
    # This is simpler than a new trainer — the Lightning test call with
    # ckpt_path="best" loads the checkpoint internally; we then apply SWA
    # weights by hooking into on_test_start via a flag.
    # -----------------------------------------------------------------------
    if fast_dev_run or args.debug_max_step is not None:
        # Quick debug mode: use current model state
        test_results = trainer.test(model, datamodule=dm)
    else:
        swa_count = model._swa.n_models if model._swa.is_active() else 0
        if swa_count >= 3:
            print(f"[Node4-3-2-1-2] SWA active with {swa_count} averaged models.")
            # Mark model so on_test_start can apply SWA weights
            model._swa_apply_at_test_start = True
            test_results = trainer.test(model, datamodule=dm, ckpt_path="best")
        else:
            print(f"[Node4-3-2-1-2] SWA has {swa_count} models (need ≥3). "
                  f"Using single best checkpoint.")
            test_results = trainer.test(model, datamodule=dm, ckpt_path="best")

    print(f"[Node4-3-2-1-2] Training complete. Test results: {test_results}")


if __name__ == "__main__":
    main()
