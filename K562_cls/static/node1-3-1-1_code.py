"""Node 1-3-2: Fine-tuned scFoundation (top-6 layers) + Frozen STRING_GNN + GatedFusion + GenePriorBias.

Strategy: Depart from the frozen-embedding fusion path (confirmed dead end across 3 nodes).
Adopt node4-2's proven fine-tuned scFoundation approach with:
  1. Fine-tune scFoundation top-6 layers (not frozen) — addresses root cause: frozen scFoundation
     cannot encode perturbation-specific responses
  2. Simpler GatedFusion (concatenation-based, not EnhancedGatedFusion with residual)
  3. 2-layer MLP head (512→256→19920) matching node4-2's proven design
  4. GenePriorBias [3, 6640] from node4-2-1-1 (+0.0035 F1)
  5. Mixup alpha=0.2 at fusion embedding level (regularizer that worked in node4-2)
  6. FIX: min_lr_ratio 0.10→0.05 (node4-2-1-1 oscillated at epoch 248+; feedback says lower floor)
  7. Discriminative LR: scFoundation FT layers=1e-4, fusion+head=3e-4
  8. Extended training: max_epochs=300, patience=30

Memory connections:
- node4-2 (F1=0.4801): base architecture — fine-tuned scFoundation is the key
- node4-2-1-1 (F1=0.4836): GenePriorBias works; min_lr_ratio=0.10 caused oscillation
- node1-3-1 (F1=0.4689, parent): frozen scFoundation confirmed as dead end
- node1-3 feedback: "abandon frozen-embedding fusion path entirely"
- Feedback: "pursue node4-2 fine-tuned scFoundation (highest priority — targets F1=0.480+)"
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640
N_CLASSES = 3
# Remapped class frequencies (after -1→0, 0→1, 1→2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

STRING_GNN_DIR   = Path("/home/Models/STRING_GNN")
SCFOUNDATION_DIR = Path("/home/Models/scFoundation")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights; neutral class stays ≈ 1."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0,1,2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)          # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)          # [N, G]
        is_pred = (y_hat == c)            # [N, G]
        present = is_true.any(dim=0)      # [G]

        tp = (is_pred & is_true).float().sum(0)
        fp = (is_pred & ~is_true).float().sum(0)
        fn = (~is_pred & is_true).float().sum(0)

        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(
            prec + rec > 0,
            2 * prec * rec / (prec + rec + 1e-8),
            torch.zeros_like(prec),
        )

        f1_per_gene += f1_c * present.float()
        n_present   += present.float()

    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# Pre-computation utilities
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_string_gnn_embeddings() -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load STRING_GNN and compute all node embeddings. Returns (emb[N,256], pert_id→idx)."""
    model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
    model.eval()
    graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
    node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())

    edge_index = graph["edge_index"]
    ew = graph.get("edge_weight", None)

    outputs = model(edge_index=edge_index, edge_weight=ew)
    emb = outputs.last_hidden_state.float().cpu()   # [18870, 256]

    pert_to_idx = {name: i for i, name in enumerate(node_names)}
    del model
    return emb, pert_to_idx


# ---------------------------------------------------------------------------
# GatedFusion Module (simpler, concatenation-based — matches node4-2 design)
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Gated fusion: cat[scf(768), gnn(256)] → [out_dim=512].

    Unlike node1-3-1's EnhancedGatedFusion with residual bypasses, this is
    a cleaner design appropriate for the fine-tuned scFoundation setting where
    the backbone already adapts to the task (no need for GNN residual safety net).
    """

    def __init__(
        self,
        scf_dim: int = 768,
        gnn_dim: int = 256,
        out_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        in_dim = scf_dim + gnn_dim  # 1024
        self.proj = nn.Linear(in_dim, out_dim * 2)  # gate half + value half
        self.out_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, scf_emb: torch.Tensor, gnn_emb: torch.Tensor) -> torch.Tensor:
        """scf_emb [B, 768], gnn_emb [B, 256] → [B, out_dim]."""
        x = torch.cat([scf_emb, gnn_emb], dim=-1)  # [B, 1024]
        proj_out = self.proj(x)                     # [B, out_dim*2]
        half = proj_out.shape[-1] // 2
        gate = torch.sigmoid(proj_out[:, :half])    # [B, out_dim]
        value = proj_out[:, half:]                  # [B, out_dim]
        fused = gate * value                        # [B, out_dim]
        return self.dropout(self.out_norm(fused))


# ---------------------------------------------------------------------------
# GenePriorBias Module (from node4-2-1-1, +0.0035 F1)
# ---------------------------------------------------------------------------
class GenePriorBias(nn.Module):
    """Gene-specific class prior bias [3, 6640] initialized from class log-frequencies.

    Each gene gets a per-class bias initialized from log(CLASS_FREQ), so predictions
    start with the right class priors before fine-tuning. This allows the model to
    focus capacity on learning per-gene deviations from the global prior.
    Proven to provide +0.0035 test F1 in node4-2-1-1.
    """

    def __init__(self, n_classes: int = N_CLASSES, n_genes: int = N_GENES) -> None:
        super().__init__()
        # Initialize from log class frequencies: [3] → broadcast to [3, G]
        log_freq = torch.tensor(
            [float(np.log(f + 1e-9)) for f in CLASS_FREQ], dtype=torch.float32
        )
        init_bias = log_freq.unsqueeze(1).expand(-1, n_genes).clone()  # [3, 6640]
        self.bias = nn.Parameter(init_bias)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: [B, 3, G] → [B, 3, G] with gene-specific class priors added."""
        return logits + self.bias.unsqueeze(0)   # [1, 3, G] broadcasts over B


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    """K562 DEG prediction dataset."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        self.sample_indices = list(range(len(df)))
        has_label = "label" in df.columns and df["label"].notna().all()
        if has_label:
            self.labels = [
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
            item["labels"] = self.labels[idx]   # [G] in {0,1,2}
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_idx": torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
        "pert_id":    [b["pert_id"]  for b in batch],
        "symbol":     [b["symbol"]   for b in batch],
    }
    if "labels" in batch[0]:
        out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df)
        self.val_ds   = DEGDataset(val_df)
        self.test_ds  = DEGDataset(test_df)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Lightning Model
# ---------------------------------------------------------------------------
class ScfGnnFusionDEGModel(pl.LightningModule):
    """Fine-tuned scFoundation (top N layers) + Frozen STRING_GNN + GatedFusion + GenePriorBias.

    Architecture pipeline:
      pert_id
        |
        +--- scFoundation (top-6 layers fine-tuned, layers 6-11; bottom-6 frozen)
        |    Encode as {gene_ids:[pert_id], expression:[1.0]} → [B, 3, 768] → mean→ [B, 768]
        |
        +--- STRING_GNN (FROZEN, pre-computed buffer [18870, 256])
        |    Direct lookup: [B, 256]
        |
        GatedFusion(768+256→1024→512): cat → proj → gate×value → LayerNorm → Dropout
             |
        [Optional Mixup at this level during training]
             |
        Head: LayerNorm(512) → Linear(512→256) → GELU → Dropout(0.5) → Linear(256→19920)
             |
        reshape [B, 3, 6640]
             |
        GenePriorBias (learnable [3, 6640], initialized from log-class-frequencies)
             |
        Output: [B, 3, 6640] logits → softmax probabilities
    """

    def __init__(
        self,
        scf_ft_layers: int = 6,
        fusion_out_dim: int = 512,
        head_hidden_dim: int = 256,
        fusion_dropout: float = 0.2,
        head_dropout: float = 0.5,
        scf_lr: float = 1e-4,
        head_lr: float = 3e-4,
        weight_decay: float = 2e-2,
        warmup_epochs: int = 10,
        t_max: int = 290,
        min_lr_ratio: float = 0.05,
        label_smoothing: float = 0.1,
        mixup_alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # ---- Collect all pert_ids across splits ----
        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")
        all_pert_ids = (
            train_df["pert_id"].tolist() +
            val_df["pert_id"].tolist() +
            test_df["pert_id"].tolist()
        )
        unique_sorted = sorted(set(all_pert_ids))
        self.pert_to_pos = {pid: i for i, pid in enumerate(unique_sorted)}
        M = len(unique_sorted)

        # ---- Load scFoundation tokenizer (rank-0 first, barrier) ----
        if local_rank == 0:
            AutoTokenizer.from_pretrained(str(SCFOUNDATION_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(str(SCFOUNDATION_DIR), trust_remote_code=True)

        # ---- Pre-tokenize all pert_ids (one-time CPU op, cached as non-persistent buffer) ----
        self.print(f"Pre-tokenizing {M} unique pert_ids for scFoundation...")
        all_input_ids_list = []
        for start in range(0, M, 128):
            batch_pids = unique_sorted[start:start + 128]
            expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in batch_pids]
            tok_out = tokenizer(expr_dicts, return_tensors="pt")
            all_input_ids_list.append(tok_out["input_ids"])   # [B, 19264] float32
        scf_input_tensor = torch.cat(all_input_ids_list, dim=0)  # [M, 19264] float32
        # persistent=False: not saved in checkpoint (recomputed in setup each time)
        self.register_buffer("scf_input_ids", scf_input_tensor, persistent=False)

        # ---- Load scFoundation model + fine-tune top N layers ----
        self.print(f"Loading scFoundation (fine-tuning top {hp.scf_ft_layers} layers)...")
        scf_model = AutoModel.from_pretrained(str(SCFOUNDATION_DIR), trust_remote_code=True)
        scf_model.config.use_cache = False
        scf_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Freeze all parameters first
        for p in scf_model.parameters():
            p.requires_grad = False

        # Unfreeze top scf_ft_layers transformer layers (layers 6-11 for ft_layers=6)
        n_total_layers = 12
        ft_start = n_total_layers - hp.scf_ft_layers  # = 6
        for layer_idx in range(ft_start, n_total_layers):
            for p in scf_model.encoder.transformer_encoder[layer_idx].parameters():
                p.requires_grad = True
        # Also unfreeze the final encoder LayerNorm
        for p in scf_model.encoder.norm.parameters():
            p.requires_grad = True

        # Cast fine-tuned params to float32 for stable optimization
        for p in scf_model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        self.scf_model = scf_model
        scf_ft_param_count = sum(p.numel() for p in scf_model.parameters() if p.requires_grad)
        self.print(f"scFoundation trainable params: {scf_ft_param_count:,} "
                   f"(layers {ft_start}-{n_total_layers - 1} + final LN)")

        # ---- STRING_GNN: pre-compute embeddings (fully frozen, non-persistent buffer) ----
        self.print("Precomputing STRING_GNN embeddings (frozen)...")
        string_emb, pert_to_gnn_idx = precompute_string_gnn_embeddings()
        # non-persistent: recomputed in setup, not saved in checkpoint
        self.register_buffer("node_embeddings", string_emb, persistent=False)   # [18870, 256]

        # Build pert_id → STRING_GNN node index mapping
        gnn_idx_tensor = torch.tensor(
            [pert_to_gnn_idx.get(pid, -1) for pid in unique_sorted], dtype=torch.long
        )
        self.register_buffer("pert_gnn_idx", gnn_idx_tensor, persistent=False)  # [M]

        # Fallback embedding for pert_ids not in STRING_GNN
        self.fallback_emb = nn.Parameter(torch.zeros(1, 256))

        # ---- Trainable modules: GatedFusion + 2-layer head + GenePriorBias ----
        self.fusion = GatedFusion(
            scf_dim=768,
            gnn_dim=256,
            out_dim=hp.fusion_out_dim,
            dropout=hp.fusion_dropout,
        )

        # 2-layer classification head: fusion_out_dim → head_hidden_dim → N_CLASSES*N_GENES
        self.head = nn.Sequential(
            nn.LayerNorm(hp.fusion_out_dim),
            nn.Linear(hp.fusion_out_dim, hp.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden_dim, N_CLASSES * N_GENES),
        )

        # GenePriorBias: learnable [3, 6640] bias from class log-frequencies
        self.gene_prior_bias = GenePriorBias(n_classes=N_CLASSES, n_genes=N_GENES)

        self.register_buffer("class_weights", get_class_weights())

        # Cast all remaining trainable parameters to float32
        for name, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Accumulators for val/test
        self._val_preds: List[torch.Tensor] = []
        self._val_tgts:  List[torch.Tensor] = []
        self._val_idx:   List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    # ---- Embedding extractors ----
    def _get_scf_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Fine-tuned scFoundation embedding [B, 768]."""
        pos = torch.tensor(
            [self.pert_to_pos[pid] for pid in pert_ids], dtype=torch.long, device=self.device
        )
        input_ids = self.scf_input_ids[pos]  # [B, 19264] float32
        outputs = self.scf_model(input_ids=input_ids, output_hidden_states=False)
        # last_hidden_state: [B, nnz+2, 768] = [B, 3, 768] for nnz=1 (single-gene)
        h = outputs.last_hidden_state.float()   # cast from bf16 to float32
        return h.mean(dim=1)                    # [B, 768]

    def _get_gnn_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Frozen STRING_GNN embedding [B, 256]."""
        pos = torch.tensor(
            [self.pert_to_pos[pid] for pid in pert_ids], dtype=torch.long, device=self.device
        )
        gnn_node_idx = self.pert_gnn_idx[pos]   # [B]
        valid = gnn_node_idx >= 0
        safe_idx = gnn_node_idx.clamp(min=0)
        emb = self.node_embeddings[safe_idx]     # [B, 256]
        fallback = self.fallback_emb.expand(emb.shape[0], -1).to(emb.dtype)
        emb = torch.where(valid.unsqueeze(-1), emb, fallback)
        return emb.float()

    # ---- Forward ----
    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        scf_emb = self._get_scf_emb(pert_ids)       # [B, 768]
        gnn_emb = self._get_gnn_emb(pert_ids)       # [B, 256]
        fused   = self.fusion(scf_emb, gnn_emb)     # [B, fusion_out_dim=512]
        raw     = self.head(fused)                   # [B, N_CLASSES * N_GENES]
        logits  = raw.reshape(-1, N_CLASSES, N_GENES)  # [B, 3, 6640]
        logits  = self.gene_prior_bias(logits)          # [B, 3, 6640] + prior
        return logits

    # ---- Loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),   # [B*G, 3]
            targets.reshape(-1),                        # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ---- Steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        pert_ids = batch["pert_id"]
        labels   = batch["labels"]  # [B, G]
        hp = self.hparams

        if hp.mixup_alpha > 0:
            # Mixup at fusion embedding level (proven effective in node4-2)
            scf_emb = self._get_scf_emb(pert_ids)  # [B, 768]
            gnn_emb = self._get_gnn_emb(pert_ids)  # [B, 256]
            fused   = self.fusion(scf_emb, gnn_emb)  # [B, 512]

            lam = float(np.random.beta(hp.mixup_alpha, hp.mixup_alpha))
            idx = torch.randperm(fused.shape[0], device=fused.device)
            fused_mix = lam * fused + (1 - lam) * fused[idx]

            raw    = self.head(fused_mix)                 # [B, N_CLASSES * N_GENES]
            logits = raw.reshape(-1, N_CLASSES, N_GENES)  # [B, 3, 6640]
            logits = self.gene_prior_bias(logits)

            loss = lam * self._loss(logits, labels) + (1 - lam) * self._loss(logits, labels[idx])
        else:
            logits = self(pert_ids)
            loss   = self._loss(logits, labels)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits, dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, dim=0)    # [N_local, 3, G]
        local_tgts  = torch.cat(self._val_tgts,  dim=0)    # [N_local, G]
        local_idx   = torch.cat(self._val_idx,   dim=0)    # [N_local]
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds)   # [W, N_local, 3, G]
        all_tgts  = self.all_gather(local_tgts)    # [W, N_local, G]
        all_idx   = self.all_gather(local_idx)     # [W, N_local]

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

        order  = torch.argsort(idx_flat)
        s_idx  = idx_flat[order]
        s_pred = preds_flat[order]
        s_tgt  = tgts_flat[order]
        mask   = torch.cat([torch.ones(1, dtype=torch.bool, device=s_idx.device),
                            s_idx[1:] != s_idx[:-1]])
        preds_dedup = s_pred[mask]
        tgts_dedup  = s_tgt[mask]

        f1 = compute_per_gene_f1(preds_dedup, tgts_dedup)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        probs  = torch.softmax(logits, dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, dim=0)    # [N_local, 3, G]
        local_idx   = torch.cat(self._test_idx,   dim=0)    # [N_local]
        all_preds   = self.all_gather(local_preds)           # [W, N_local, 3, G]
        all_idx     = self.all_gather(local_idx)             # [W, N_local]

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            order = torch.argsort(idx_flat)
            s_idx = idx_flat[order]
            s_pred = preds_flat[order]
            mask = torch.cat([torch.ones(1, dtype=torch.bool, device=s_idx.device),
                              s_idx[1:] != s_idx[:-1]])
            preds_dedup = s_pred[mask]         # [N_test, 3, G]
            unique_sid  = s_idx[mask].tolist()

            test_df = pd.read_csv(TEST_TSV, sep="\t")
            idx_to_meta = {i: (test_df.iloc[i]["pert_id"], test_df.iloc[i]["symbol"])
                           for i in range(len(test_df))}

            rows = []
            for sid in unique_sid:
                pid, sym = idx_to_meta[int(sid)]
                dedup_pos = (s_idx == sid).nonzero(as_tuple=True)[0][0].item()
                pred_list = preds_dedup[dedup_pos].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            self.print(f"[Node1-3-2] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()

    # ---- Checkpoint helpers ----
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and persistent buffers."""
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
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        bufs  = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {train}/{total} params "
            f"({100 * train / total:.2f}%), plus {bufs} buffer values"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        """Load partial checkpoint (trainable params + buffers only)."""
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer (Discriminative LR) ----
    def configure_optimizers(self):
        hp = self.hparams

        # Group 1: scFoundation fine-tuned layers (lower LR — careful adaptation)
        scf_params = [p for p in self.scf_model.parameters() if p.requires_grad]

        # Group 2: fusion + head + gene_prior_bias + fallback_emb (higher LR — learns from scratch)
        other_params = (
            list(self.fusion.parameters()) +
            list(self.head.parameters()) +
            list(self.gene_prior_bias.parameters()) +
            [self.fallback_emb]
        )

        param_groups = [
            {"params": scf_params,   "lr": hp.scf_lr,  "name": "scf_backbone"},
            {"params": other_params, "lr": hp.head_lr, "name": "fusion_head"},
        ]

        opt = torch.optim.AdamW(param_groups, weight_decay=hp.weight_decay)

        # LambdaLR: linear warmup + cosine decay with min_lr_ratio floor
        # Same multiplicative schedule for all groups, preserving the LR ratio.
        # At convergence: scf_lr → scf_lr * min_lr_ratio, head_lr → head_lr * min_lr_ratio
        def lr_lambda(epoch: int) -> float:
            if epoch < hp.warmup_epochs:
                # Linear warmup from 10% to 100% of target LR
                return 0.1 + 0.9 * (epoch / max(1, hp.warmup_epochs))
            cos_epoch = epoch - hp.warmup_epochs
            # Cosine decay for t_max epochs, then stays at floor
            cos_progress = min(float(cos_epoch), float(hp.t_max)) / float(hp.t_max)
            cos_val = 0.5 * (1.0 + np.cos(np.pi * cos_progress))
            # Scale between min_lr_ratio and 1.0
            return hp.min_lr_ratio + (1.0 - hp.min_lr_ratio) * cos_val

        # Apply same lambda to both groups (ratio scf_lr/head_lr preserved)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lr_lambda, lr_lambda])

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node1-3-2: Fine-tuned scFoundation + Frozen STRING_GNN + GatedFusion + GenePriorBias"
    )
    parser.add_argument("--micro-batch-size",   type=int,   default=32)
    parser.add_argument("--global-batch-size",  type=int,   default=256)
    parser.add_argument("--max-epochs",         type=int,   default=300)
    parser.add_argument("--scf-lr",             type=float, default=1e-4,
                        help="LR for scFoundation fine-tuned layers (top-6)")
    parser.add_argument("--head-lr",            type=float, default=3e-4,
                        help="LR for fusion+head+GenePriorBias (3x higher than scf_lr)")
    parser.add_argument("--weight-decay",       type=float, default=2e-2)
    parser.add_argument("--scf-ft-layers",      type=int,   default=6,
                        help="Number of top scFoundation layers to fine-tune (0-12)")
    parser.add_argument("--fusion-out-dim",     type=int,   default=512)
    parser.add_argument("--head-hidden-dim",    type=int,   default=256)
    parser.add_argument("--fusion-dropout",     type=float, default=0.2)
    parser.add_argument("--head-dropout",       type=float, default=0.5)
    parser.add_argument("--warmup-epochs",      type=int,   default=10)
    parser.add_argument("--t-max",              type=int,   default=290,
                        help="Cosine decay epochs (after warmup); warmup+t_max=max_epochs")
    parser.add_argument("--min-lr-ratio",       type=float, default=0.05,
                        help="Floor LR ratio; fixes node4-2-1-1's oscillation at 0.10")
    parser.add_argument("--label-smoothing",    type=float, default=0.1)
    parser.add_argument("--mixup-alpha",        type=float, default=0.2,
                        help="Mixup alpha at fusion embedding level (0=disabled)")
    parser.add_argument("--patience",           type=int,   default=30)
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--num-workers",        type=int,   default=4)
    parser.add_argument("--debug-max-step",     type=int,   default=None, dest="debug_max_step")
    parser.add_argument("--fast-dev-run",       action="store_true", dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = args.debug_max_step
        lim_val   = args.debug_max_step
        lim_test  = 1.0   # All test data (154 samples, fast)
        max_steps = args.debug_max_step
    else:
        lim_train = 1.0
        lim_val   = 1.0
        lim_test  = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # DataModule + Model
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    model = ScfGnnFusionDEGModel(
        scf_ft_layers   = args.scf_ft_layers,
        fusion_out_dim  = args.fusion_out_dim,
        head_hidden_dim = args.head_hidden_dim,
        fusion_dropout  = args.fusion_dropout,
        head_dropout    = args.head_dropout,
        scf_lr          = args.scf_lr,
        head_lr         = args.head_lr,
        weight_decay    = args.weight_decay,
        warmup_epochs   = args.warmup_epochs,
        t_max           = args.t_max,
        min_lr_ratio    = args.min_lr_ratio,
        label_smoothing = args.label_smoothing,
        mixup_alpha     = args.mixup_alpha,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    early_stop_callback = EarlyStopping(
        monitor="val/f1",
        mode="max",
        patience=args.patience,
        min_delta=1e-4,
        verbose=True,
    )
    lr_monitor   = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=10)

    csv_logger         = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tensorboard_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        # find_unused_parameters=True needed for fallback_emb (may not appear in every batch)
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accum,
        limit_train_batches=lim_train,
        limit_val_batches=lim_val,
        limit_test_batches=lim_test,
        val_check_interval=args.val_check_interval if (args.debug_max_step is None and not fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, progress_bar],
        logger=[csv_logger, tensorboard_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=dm)
    else:
        test_results = trainer.test(model, datamodule=dm, ckpt_path="best")

    # Save test results
    if test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        with open(score_path, "w") as f:
            f.write(str(test_results))
        print(f"[Node1-3-2] Test results: {test_results}")


if __name__ == "__main__":
    main()
