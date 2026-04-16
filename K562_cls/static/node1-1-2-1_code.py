"""Node 1-1-2-1 – Triple Backbone (STRING_GNN + ESM2-650M + scFoundation) + GatedFusion
                + GenePriorBias + Bilinear Head + Mixup + SWA

Strategy:
- Parent (node1-1-2, F1=0.4585) used STRING_GNN+ESM2-650M with asymmetric focal loss.
  The core bottleneck was that both backbones encode static, non-perturbation-aware
  representations. This node directly addresses that by adding scFoundation as a
  perturbation-aware third backbone.

- Three orthogonal biological representations:
  1. STRING_GNN (frozen, pre-computed, 256-dim): PPI network topology
  2. ESM2-650M (frozen, pre-computed, 3840→256 projected): protein sequence/structure
  3. scFoundation (top-6 layers fine-tuned, 768-dim): single-cell transcriptomic context
     encoded via perturbation-identity input (expression=1.0 for target gene, 0.0 elsewhere)

- Architecture pipeline:
  - Static backbone: STRING_GNN[B,256] + ESM2_proj[B,256] → concat → LayerNorm → [B,512]
  - scFoundation: mean-pool last_hidden_state → [B,768]
  - GatedFusion(512+768→512): element-wise gates from concatenated input
  - Head projection: 512 → 256 (bilinear_dim) via dropout+linear+LayerNorm+GELU
  - Bilinear head: h[B,256] @ gene_class_emb[3,G,256] / sqrt(256) → [B,3,G]
  - GenePriorBias[3,G]: per-gene class priors, gradient zeroed for first 50 epochs

- Loss: Weighted CE + label smoothing=0.1 (better F1 alignment than focal)
- Mixup: alpha=0.2 on fused embeddings (proven in node4-2 lineage)
- SWA from epoch 180 (proven +0.003 in node4-2-1-1-1)
- WarmupCosine LR (min_lr_ratio=0.05) with 10-epoch warmup
- lr=2e-4, weight_decay=3e-2, patience=25, max_epochs=300

Key differences from parent (node1-1-2, F1=0.4585):
- ADDS scFoundation top-6 layers fine-tuned (perturbation-aware dynamic repr.)
- REPLACES asymmetric focal loss with weighted CE + label smoothing (F1-aligned)
- REPLACES CAWR with WarmupCosine (min_lr_ratio=0.05) - proven better
- ADDS GenePriorBias with delayed unfreezing (gradient zeroed for first 50 epochs)
- ADDS Mixup augmentation (alpha=0.2)
- ADDS SWA from epoch 180
- ESM2 projected to 256 before concatenation with STRING_GNN (cleaner fusion)
- GatedFusion for scFoundation integration (instead of plain concat+projection)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
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
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES     = 6640
N_CLASSES   = 3

SCF_HIDDEN  = 768    # scFoundation hidden dimension
GNN_HIDDEN  = 256    # STRING_GNN hidden dimension
ESM2_DIM    = 3840   # ESM2-650M embedding dimension
ESM2_PROJ   = 256    # ESM2 projected dimension (to match STRING_GNN dim)
STATIC_DIM  = GNN_HIDDEN + ESM2_PROJ  # 512: concatenated STRING+ESM2(projected)
FUSION_DIM  = 512    # GatedFusion output dimension
BILINEAR_DIM = 256   # Bilinear head embedding dimension

SCF_MODEL_DIR = "/home/Models/scFoundation"
GNN_MODEL_DIR = Path("/home/Models/STRING_GNN")

# Remapped class frequencies: class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Inverse-sqrt-frequency class weights for 92.5% neutral class imbalance."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float – softmax probabilities
        targets: [N, G]    long  – class labels in {0,1,2}
    Returns:
        Scalar float: mean F1 over all G genes.
    """
    y_hat = preds.argmax(dim=1)  # [N, G]
    G = targets.shape[1]
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
            prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-8), torch.zeros_like(prec)
        )
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Element-wise gated fusion of static backbone (512) and scFoundation (768).

    Architecture from node4-2 lineage (best in tree at 0.4868):
    output = LayerNorm(Dropout(gate_static * proj_static(static) + gate_scf * proj_scf(scf)))
    """

    def __init__(
        self,
        d_static: int = STATIC_DIM,
        d_scf: int = SCF_HIDDEN,
        d_out: int = FUSION_DIM,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        d_in = d_static + d_scf
        self.proj_static = nn.Linear(d_static, d_out)
        self.proj_scf    = nn.Linear(d_scf, d_out)
        self.gate_static = nn.Linear(d_in, d_out)
        self.gate_scf    = nn.Linear(d_in, d_out)
        self.layer_norm  = nn.LayerNorm(d_out)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, static_emb: torch.Tensor, scf_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            static_emb: [B, STATIC_DIM=512] STRING_GNN+ESM2(projected) combined
            scf_emb:    [B, SCF_HIDDEN=768] scFoundation mean-pooled
        Returns:
            [B, FUSION_DIM=512] fused representation
        """
        combined   = torch.cat([static_emb, scf_emb], dim=-1)        # [B, 1280]
        gate_s     = torch.sigmoid(self.gate_static(combined))        # [B, 512]
        gate_f     = torch.sigmoid(self.gate_scf(combined))           # [B, 512]
        fused      = gate_s * self.proj_static(static_emb) + gate_f * self.proj_scf(scf_emb)
        return self.dropout(self.layer_norm(fused))                   # [B, 512]


# ---------------------------------------------------------------------------
# SWA Callback
# ---------------------------------------------------------------------------
class SWACallback(pl.Callback):
    """Stochastic Weight Averaging callback.

    Averages model weights from swa_start_epoch onward. Saves SWA state to disk
    in on_train_end so it survives Lightning's best-checkpoint reload during
    on_test_start (before test epochs run).

    Proven +0.0032 improvement in node4-2-1-1-1 lineage (best in tree).
    """

    def __init__(self, swa_start_epoch: int = 180) -> None:
        super().__init__()
        self.swa_start_epoch = swa_start_epoch
        self._swa_model: Optional[AveragedModel] = None
        self._n_averaged: int = 0
        self._swa_state_path: Optional[Path] = None

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: "TripleBackboneBilinearModel"
    ) -> None:
        """Update SWA model at end of each epoch in SWA phase."""
        current_epoch = trainer.current_epoch
        if current_epoch < self.swa_start_epoch:
            return
        if self._swa_model is None:
            self._swa_model = AveragedModel(pl_module)
            pl_module.print(
                f"[SWACallback] Epoch {current_epoch}: Initialized SWA model averaging"
            )
        self._swa_model.update_parameters(pl_module)
        self._n_averaged += 1
        if self._n_averaged % 20 == 0:
            pl_module.print(
                f"[SWACallback] Epoch {current_epoch}: SWA averaged {self._n_averaged} checkpoints"
            )

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: "TripleBackboneBilinearModel"
    ) -> None:
        """Save SWA-averaged state to disk (persists across Lightning's checkpoint reload)."""
        if self._swa_model is None or self._n_averaged == 0:
            pl_module.print("[SWACallback] No SWA averaging performed.")
            return

        pl_module.print(
            f"[SWACallback] Saving SWA state averaged over "
            f"{self._n_averaged} checkpoints from epoch {self.swa_start_epoch}."
        )

        # Move SWA model to same device as pl_module
        device = next(pl_module.parameters()).device
        self._swa_model = self._swa_model.to(device)

        # Build SWA state dict (params + buffers), casting to float32 for storage
        swa_state: Dict[str, torch.Tensor] = {}
        for name, param in self._swa_model.module.named_parameters():
            swa_state[name] = param.data.clone().float()
        for name, buf in self._swa_model.module.named_buffers():
            swa_state[name] = buf.clone().float()

        # Persist to disk so it survives Lightning's best-checkpoint reload in on_test_start
        output_dir = Path(__file__).parent / "run"
        output_dir.mkdir(parents=True, exist_ok=True)
        self._swa_state_path = output_dir / "swa_state.pt"
        torch.save(swa_state, self._swa_state_path)
        pl_module.print(f"[SWACallback] SWA state saved to {self._swa_state_path}.")
        pl_module._swa_applied = True

    def on_test_start(
        self, trainer: pl.Trainer, pl_module: "TripleBackboneBilinearModel"
    ) -> None:
        """Reload SWA weights after Lightning's best-checkpoint restore (before test epochs)."""
        if not pl_module._swa_applied or self._swa_state_path is None:
            return
        if not self._swa_state_path.exists():
            pl_module.print(
                f"[SWACallback] WARNING: SWA state not found at {self._swa_state_path}. "
                "SWA weights may have been overwritten by best checkpoint."
            )
            return

        swa_state = torch.load(self._swa_state_path, map_location="cpu")
        loaded = 0
        skipped = 0
        for name, param in pl_module.named_parameters():
            if name in swa_state:
                param.data.copy_(swa_state[name].to(param.device))
                loaded += 1
            else:
                skipped += 1
        for name, buf in pl_module.named_buffers():
            if name in swa_state:
                buf.copy_(swa_state[name].to(buf.device))
        pl_module.print(
            f"[SWACallback] SWA weights restored ({loaded} params loaded, "
            f"{skipped} skipped) for test inference."
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    """K562 DEG prediction dataset."""

    def __init__(self, df: pd.DataFrame, string_map: Dict[str, int]) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        self.string_node_indices = torch.tensor(
            [string_map.get(p, -1) for p in self.pert_ids], dtype=torch.long
        )
        has_label = "label" in df.columns and df["label"].notna().all()
        if has_label:
            self.labels: Optional[List[torch.Tensor]] = [
                torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
                for row in df["label"].tolist()
            ]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "sample_idx":      idx,
            "pert_id":         self.pert_ids[idx],
            "symbol":          self.symbols[idx],
            "string_node_idx": self.string_node_indices[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def make_collate_fn(tokenizer):
    """Collate function tokenizing inputs for scFoundation.

    scFoundation strategy: perturbed gene represented with expression=1.0,
    all other genes=0.0. This provides perturbation-identity representation
    as used in the proven node4 lineage (F1=0.4868).
    """
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        out: Dict[str, Any] = {
            "sample_idx":      torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":         pert_ids,
            "symbol":          [b["symbol"] for b in batch],
            "string_node_idx": torch.stack([b["string_node_idx"] for b in batch]),
            "input_ids":       tokenized["input_ids"],
            "attention_mask":  tokenized["attention_mask"],
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
        self.tokenizer: Optional[Any]  = None
        self.string_map: Optional[Dict[str, int]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Rank-0 downloads tokenizer first to avoid race conditions
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(SCF_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(SCF_MODEL_DIR, trust_remote_code=True)

        if self.string_map is None:
            node_names: List[str] = json.loads((GNN_MODEL_DIR / "node_names.json").read_text())
            self.string_map = {name: idx for idx, name in enumerate(node_names)}

        self.train_ds = DEGDataset(pd.read_csv(TRAIN_TSV, sep="\t"), self.string_map)
        self.val_ds   = DEGDataset(pd.read_csv(VAL_TSV,   sep="\t"), self.string_map)
        self.test_ds  = DEGDataset(pd.read_csv(TEST_TSV,  sep="\t"), self.string_map)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_collate_fn(self.tokenizer),
        )

    def train_dataloader(self) -> DataLoader: return self._loader(self.train_ds, True)
    def val_dataloader(self)   -> DataLoader: return self._loader(self.val_ds,   False)
    def test_dataloader(self)  -> DataLoader:
        sampler = SequentialSampler(self.test_ds)
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_collate_fn(self.tokenizer),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TripleBackboneBilinearModel(pl.LightningModule):
    """Triple-backbone DEG prediction model.

    Architecture:
        1. STRING_GNN (frozen, pre-computed buffer [18870,256])
        2. ESM2-650M  (frozen, pre-computed buffer [18870,3840]) → linear → [B,256]
        3. Concat → LayerNorm → [B, 512] (static backbone)
        4. scFoundation (top-6 layers fine-tuned) → mean-pool → [B, 768]
        5. GatedFusion(512+768→512) → [B, 512]
        6. head_proj: Dropout → Linear(512→256) → LayerNorm → GELU → Dropout → [B, 256]
        7. Bilinear: einsum("bd,cgd->bcg", h, gene_class_emb) / sqrt(256) → [B, 3, G]
        8. GenePriorBias[3, G] added (gradient zeroed for first bias_warmup_epochs)
        9. Loss: weighted CE + label smoothing = 0.1
       10. Mixup(alpha=0.2) on fused embeddings (training only)
       11. SWA from swa_start_epoch (via SWACallback)
    """

    def __init__(
        self,
        scf_finetune_layers: int   = 6,
        head_dropout: float        = 0.5,
        fusion_dropout: float      = 0.3,
        esm2_dropout: float        = 0.2,
        bilinear_dim: int          = 256,
        lr: float                  = 2e-4,
        weight_decay: float        = 3e-2,
        warmup_epochs: int         = 10,
        max_epochs: int            = 300,
        min_lr_ratio: float        = 0.05,
        mixup_alpha: float         = 0.2,
        label_smoothing: float     = 0.1,
        bias_warmup_epochs: int    = 50,
        swa_start_epoch: int       = 180,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._swa_applied: bool = False

    def setup(self, stage: Optional[str] = None) -> None:
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True
        hp = self.hparams

        # ── scFoundation backbone (top-k layers fine-tuned) ──────────────────
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoModel.from_pretrained(SCF_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.scf = AutoModel.from_pretrained(
            SCF_MODEL_DIR,
            trust_remote_code=True,
            _use_flash_attention_2=True,
        ).to(torch.bfloat16)
        self.scf.config.use_cache = False
        self.scf.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Freeze all scFoundation params, then unfreeze top-k transformer layers + final LN
        for param in self.scf.parameters():
            param.requires_grad = False
        n_scf_layers = len(self.scf.encoder.transformer_encoder)
        for i in range(n_scf_layers - hp.scf_finetune_layers, n_scf_layers):
            for param in self.scf.encoder.transformer_encoder[i].parameters():
                param.requires_grad = True
        for param in self.scf.encoder.norm.parameters():
            param.requires_grad = True
        # Cast trainable scFoundation params to float32 for stable optimization
        for name, param in self.scf.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        scf_train = sum(p.numel() for p in self.scf.parameters() if p.requires_grad)
        scf_total = sum(p.numel() for p in self.scf.parameters())
        print(f"[Node1-1-2-1] scFoundation: {scf_train:,}/{scf_total:,} trainable params")

        # ── STRING_GNN: frozen, pre-computed embeddings ──────────────────────
        print("[Node1-1-2-1] Precomputing STRING_GNN embeddings (frozen)...")
        if local_rank == 0:
            AutoModel.from_pretrained(str(GNN_MODEL_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        gnn_temp = AutoModel.from_pretrained(str(GNN_MODEL_DIR), trust_remote_code=True).float()
        gnn_temp.eval()
        graph_data  = torch.load(GNN_MODEL_DIR / "graph_data.pt", map_location="cpu")
        edge_index  = graph_data["edge_index"].long()
        edge_weight = graph_data.get("edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.float()
        with torch.no_grad():
            gnn_out  = gnn_temp(edge_index=edge_index, edge_weight=edge_weight)
            gnn_embs = gnn_out.last_hidden_state.float().detach()  # [18870, 256]
        self.register_buffer("string_emb_table", gnn_embs)
        del gnn_temp
        print(f"[Node1-1-2-1] STRING_GNN embeddings cached: {gnn_embs.shape}")

        # ── ESM2-650M: frozen, pre-computed embeddings ───────────────────────
        esm2_emb_table = torch.load(
            GNN_MODEL_DIR / "esm2_embeddings_t33_650M.pt", map_location="cpu"
        ).float()  # [18870, 3840]
        self.register_buffer("esm2_emb_table", esm2_emb_table)
        print(f"[Node1-1-2-1] ESM2-650M embeddings cached: {esm2_emb_table.shape}")

        # ── Fallback learnable embeddings for unknown Ensembl IDs ────────────
        self.fallback_string = nn.Embedding(1, GNN_HIDDEN)
        self.fallback_esm2   = nn.Embedding(1, ESM2_DIM)
        nn.init.normal_(self.fallback_string.weight, std=0.02)
        nn.init.normal_(self.fallback_esm2.weight,   std=0.02)

        # ── ESM2 projection: 3840 → 256 ─────────────────────────────────────
        self.esm2_proj = nn.Sequential(
            nn.LayerNorm(ESM2_DIM),
            nn.Linear(ESM2_DIM, ESM2_PROJ),
            nn.GELU(),
            nn.Dropout(hp.esm2_dropout),
        )

        # ── Static backbone normalization: concat(STRING[256]+ESM2_proj[256]) → LayerNorm ─
        self.static_norm = nn.LayerNorm(STATIC_DIM)  # LayerNorm over 512-dim

        # ── GatedFusion: static(512) + scFoundation(768) → 512 ─────────────
        self.fusion = GatedFusion(
            d_static=STATIC_DIM,
            d_scf=SCF_HIDDEN,
            d_out=FUSION_DIM,
            dropout=hp.fusion_dropout,
        )

        # ── Head projection: 512 → bilinear_dim=256 ─────────────────────────
        self.head_proj = nn.Sequential(
            nn.Dropout(hp.head_dropout),
            nn.Linear(FUSION_DIM, hp.bilinear_dim),
            nn.LayerNorm(hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.head_dropout * 0.5),
        )

        # ── Bilinear gene-class embedding [N_CLASSES, N_GENES, bilinear_dim] ─
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )

        # ── GenePriorBias [N_CLASSES, N_GENES]: per-gene class priors ────────
        # Initialize from training log-class-frequencies, zero-mean per gene
        log_freqs = torch.tensor([math.log(f + 1e-9) for f in CLASS_FREQ], dtype=torch.float32)
        gene_bias_init = log_freqs.unsqueeze(1).expand(N_CLASSES, N_GENES).clone()
        gene_bias_init = gene_bias_init - gene_bias_init.mean(0, keepdim=True)
        self.gene_prior_bias = nn.Parameter(gene_bias_init)

        # ── Loss ─────────────────────────────────────────────────────────────
        self.register_buffer("class_weights", get_class_weights())

        # ── Cast all trainable non-scf parameters to float32 ─────────────────
        for name, param in self.named_parameters():
            if param.requires_grad and not name.startswith("scf."):
                param.data = param.data.float()

        # ── Accumulators for val/test ─────────────────────────────────────────
        self._val_preds:     List[torch.Tensor] = []
        self._val_tgts:      List[torch.Tensor] = []
        self._val_idx:       List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_idx:      List[torch.Tensor] = []

    # ── Internal helper: static embedding lookup ──────────────────────────
    def _get_static_embeddings(
        self, string_node_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lookup frozen STRING_GNN and ESM2-650M embeddings.

        Args:
            string_node_idx: [B] long, -1 for unknowns
        Returns:
            string_emb: [B, 256]
            esm2_emb:   [B, 3840]
        """
        B = string_node_idx.shape[0]
        known_mask   = string_node_idx >= 0
        unknown_mask = ~known_mask
        zero_idx     = torch.zeros(
            unknown_mask.sum(), dtype=torch.long, device=string_node_idx.device
        )

        string_emb = torch.zeros(
            B, GNN_HIDDEN, dtype=torch.float32, device=string_node_idx.device
        )
        if known_mask.any():
            string_emb[known_mask] = self.string_emb_table[string_node_idx[known_mask]].float()
        if unknown_mask.any():
            string_emb[unknown_mask] = self.fallback_string(zero_idx).float()

        esm2_emb = torch.zeros(
            B, ESM2_DIM, dtype=torch.float32, device=string_node_idx.device
        )
        if known_mask.any():
            esm2_emb[known_mask] = self.esm2_emb_table[string_node_idx[known_mask]].float()
        if unknown_mask.any():
            esm2_emb[unknown_mask] = self.fallback_esm2(zero_idx).float()

        return string_emb, esm2_emb

    # ── Embedding fusion ──────────────────────────────────────────────────
    def get_fused_emb(
        self,
        input_ids:       torch.Tensor,
        attention_mask:  torch.Tensor,
        string_node_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute fused [B, FUSION_DIM=512] embedding from all three backbones."""
        # Static backbones: STRING_GNN + ESM2 → [B, 512]
        string_emb, esm2_emb = self._get_static_embeddings(string_node_idx)
        esm2_projected = self.esm2_proj(esm2_emb)                  # [B, 256]
        static_emb = self.static_norm(
            torch.cat([string_emb, esm2_projected], dim=-1)        # [B, 512]
        )

        # scFoundation: nnz=1 → output seq len=3, mean-pool → [B, 768]
        scf_out = self.scf(input_ids=input_ids, attention_mask=attention_mask)
        scf_emb = scf_out.last_hidden_state.float().mean(dim=1)    # [B, 768]

        # GatedFusion: static(512) + scFoundation(768) → [B, 512]
        return self.fusion(static_emb, scf_emb)

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids:       torch.Tensor,
        attention_mask:  torch.Tensor,
        string_node_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits [B, N_CLASSES, N_GENES]."""
        fused  = self.get_fused_emb(input_ids, attention_mask, string_node_idx)
        h      = self.head_proj(fused)                             # [B, bilinear_dim=256]

        # Scaled bilinear
        scale  = self.hparams.bilinear_dim ** 0.5
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb) / scale

        # GenePriorBias
        logits = logits + self.gene_prior_bias.unsqueeze(0)        # [B, 3, G]
        return logits

    # ── Loss ─────────────────────────────────────────────────────────────
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Weighted CE + label smoothing. Better aligned with F1 than focal loss."""
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),  # [B*G, C]
            targets.reshape(-1),                       # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ── Training step ──────────────────────────────────────────────────────
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        B      = batch["input_ids"].shape[0]
        labels = batch["labels"]

        # Always call self.forward() first so DDP sees all parameters in the loss graph.
        # Mixup is applied at the fused-embedding level and recomputes logits with the
        # mixed embedding.  By keeping the mixed computation (head_proj + gene_class_emb
        # + gene_prior_bias) inside the same autograd graph as get_fused_emb (via
        # fused), every backbone parameter participates in the backward pass.
        fused = self.get_fused_emb(
            batch["input_ids"], batch["attention_mask"], batch["string_node_idx"]
        )

        # Mixup augmentation on fused embeddings (proven in node4-2 lineage)
        if self.hparams.mixup_alpha > 0.0 and B > 1 and self.training:
            lam  = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
            perm = torch.randperm(B, device=fused.device)
            fused = lam * fused + (1 - lam) * fused[perm]
            # Re-compute head + bilinear logits with the mixed fused embedding so the
            # full gradient path goes through get_fused_emb → static_norm / fusion / scf.
            logits = self._logits_from_fused(fused)
            loss   = lam * self._loss(logits, labels) + (1 - lam) * self._loss(logits, labels[perm])
        else:
            # Use self.forward() directly so Lightning's DDP unused-parameter check
            # can trace all parameters to the loss.
            logits = self(batch["input_ids"], batch["attention_mask"], batch["string_node_idx"])
            loss   = self._loss(logits, labels)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def _logits_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        """Compute logits from fused embedding (used in mixup branch)."""
        h      = self.head_proj(fused)
        scale  = self.hparams.bilinear_dim ** 0.5
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb) / scale
        logits = logits + self.gene_prior_bias.unsqueeze(0)
        return logits

    def on_after_backward(self) -> None:
        """Zero out GenePriorBias gradients during warmup phase.

        Delayed GenePriorBias learning: bias gradients zeroed for first
        bias_warmup_epochs epochs, allowing backbone to learn stable
        representations before per-gene biases are refined.
        Proven strategy in node4-2-1-1-1 (best in tree).
        """
        if self.current_epoch < self.hparams.bias_warmup_epochs:
            if self.gene_prior_bias.grad is not None:
                self.gene_prior_bias.grad.zero_()

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.hparams.bias_warmup_epochs:
            self.print(
                f"[Node1-1-2-1] Epoch {self.current_epoch}: "
                f"GenePriorBias gradients ACTIVATED"
            )
        if self.current_epoch == self.hparams.swa_start_epoch:
            self.print(
                f"[Node1-1-2-1] Epoch {self.current_epoch}: "
                f"Entering SWA averaging phase"
            )

    # ── Validation step ────────────────────────────────────────────────────
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["string_node_idx"])
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
        local_preds = torch.cat(self._val_preds, 0)  # [N_local, 3, G]
        local_tgts  = torch.cat(self._val_tgts,  0)  # [N_local, G]
        local_idx   = torch.cat(self._val_idx,   0)  # [N_local]
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        # Gather across all DDP ranks
        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        # De-duplicate (DDP padding)
        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]
        s_pred = all_preds[order]
        s_tgt  = all_tgts[order]
        mask   = torch.cat([
            torch.ones(1, dtype=torch.bool, device=s_idx.device),
            s_idx[1:] != s_idx[:-1],
        ])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    # ── Test step ──────────────────────────────────────────────────────────
    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["string_node_idx"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)  # [N_local, 3, G]
        local_idx   = torch.cat(self._test_idx,   0)  # [N_local]
        all_preds   = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_idx     = self.all_gather(local_idx).view(-1)

        if self.trainer.is_global_zero:
            # De-duplicate
            order  = torch.argsort(all_idx)
            s_idx  = all_idx[order]
            s_pred = all_preds[order]
            mask   = torch.cat([
                torch.ones(1, dtype=torch.bool, device=s_idx.device),
                s_idx[1:] != s_idx[:-1],
            ])
            preds_dedup = s_pred[mask]
            unique_sid  = s_idx[mask].tolist()

            # Reload test.tsv metadata on rank 0
            test_df = pd.read_csv(TEST_TSV, sep="\t")
            idx_to_meta = {
                i: (test_df.iloc[i]["pert_id"], test_df.iloc[i]["symbol"])
                for i in range(len(test_df))
            }

            rows = []
            for i, sid in enumerate(unique_sid):
                pid, sym = idx_to_meta[int(sid)]
                pred_list = preds_dedup[i].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_df = pd.DataFrame(rows)
            out_df.drop_duplicates(subset=["idx"], keep="first", inplace=True)
            out_df.to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1-1-2-1] Saved {len(out_df)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()

    # ── Checkpoint helpers ─────────────────────────────────────────────────
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
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"[Node1-1-2-1] Checkpoint: {trained:,}/{total:,} params "
            f"({100 * trained / total:.2f}%)"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ── Optimizer & LR Scheduler ───────────────────────────────────────────
    def configure_optimizers(self):
        hp = self.hparams
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable_params, lr=hp.lr, weight_decay=hp.weight_decay
        )

        def lr_lambda(epoch: int) -> float:
            """Linear warmup then cosine decay with min_lr_ratio floor."""
            if epoch < hp.warmup_epochs:
                return max(1e-8, epoch / max(1, hp.warmup_epochs))
            progress = (epoch - hp.warmup_epochs) / max(1, hp.max_epochs - hp.warmup_epochs)
            progress = min(progress, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return hp.min_lr_ratio + (1.0 - hp.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node1-1-2-1 – Triple Backbone STRING_GNN+ESM2-650M+scFoundation "
                    "+ GatedFusion + GenePriorBias + SWA"
    )
    parser.add_argument("--micro-batch-size",    type=int,   default=8)
    parser.add_argument("--global-batch-size",   type=int,   default=64)
    parser.add_argument("--max-epochs",          type=int,   default=300)
    parser.add_argument("--lr",                  type=float, default=2e-4)
    parser.add_argument("--weight-decay",        type=float, default=3e-2)
    parser.add_argument("--bilinear-dim",        type=int,   default=256)
    parser.add_argument("--head-dropout",        type=float, default=0.5)
    parser.add_argument("--fusion-dropout",      type=float, default=0.3)
    parser.add_argument("--esm2-dropout",        type=float, default=0.2)
    parser.add_argument("--warmup-epochs",       type=int,   default=10)
    parser.add_argument("--min-lr-ratio",        type=float, default=0.05)
    parser.add_argument("--mixup-alpha",         type=float, default=0.2)
    parser.add_argument("--label-smoothing",     type=float, default=0.1)
    parser.add_argument("--scf-finetune-layers", type=int,   default=6,
                        dest="scf_finetune_layers")
    parser.add_argument("--bias-warmup-epochs",  type=int,   default=50)
    parser.add_argument("--swa-start-epoch",     type=int,   default=180)
    parser.add_argument("--num-workers",         type=int,   default=4)
    parser.add_argument("--debug-max-step",      type=int,   default=None, dest="debug_max_step")
    parser.add_argument("--fast-dev-run",        action="store_true", dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        # Only cap TRAINING batches; validate/test over full dataset for correct DDP metrics
        lim_train = args.debug_max_step
        lim_val   = 1.0      # validate over all batches even in debug
        max_steps = args.debug_max_step
    else:
        lim_train = 1.0
        lim_val   = 1.0
        max_steps = -1

    # Always process ALL test batches for correct DDP deduplication
    lim_test = 1.0
    val_check_interval = 1.0
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # DataModule
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)

    # Model
    model = TripleBackboneBilinearModel(
        scf_finetune_layers = args.scf_finetune_layers,
        head_dropout        = args.head_dropout,
        fusion_dropout      = args.fusion_dropout,
        esm2_dropout        = args.esm2_dropout,
        bilinear_dim        = args.bilinear_dim,
        lr                  = args.lr,
        weight_decay        = args.weight_decay,
        warmup_epochs       = args.warmup_epochs,
        max_epochs          = args.max_epochs,
        min_lr_ratio        = args.min_lr_ratio,
        mixup_alpha         = args.mixup_alpha,
        label_smoothing     = args.label_smoothing,
        bias_warmup_epochs  = args.bias_warmup_epochs,
        swa_start_epoch     = args.swa_start_epoch,
    )

    # SWA callback (only in full training runs, not debug)
    use_swa = (not fast_dev_run) and (args.debug_max_step is None)
    swa_cb  = SWACallback(swa_start_epoch=args.swa_start_epoch) if use_swa else None

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
    )
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=25, min_delta=1e-4)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    active_callbacks = [ckpt_cb, es_cb, lr_cb, pg_cb]
    if swa_cb is not None:
        active_callbacks.append(swa_cb)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy: DDP for multi-GPU training
    # find_unused_parameters=True is required because gene_prior_bias gradients are
    # zeroed in on_after_backward during bias_warmup_epochs, which can confuse DDP's
    # unused-parameter detection on early training iterations.
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
        val_check_interval      = val_check_interval,
        num_sanity_val_steps    = 2,
        callbacks               = active_callbacks,
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = 1.0,
    )

    trainer.fit(model, datamodule=dm)

    # Test inference strategy:
    # - If SWA was applied by SWACallback (on_train_end), use model directly
    # - If debug/fast_dev_run, no checkpoint loading
    # - If SWA didn't run (early stopping before swa_start_epoch), use best checkpoint
    if fast_dev_run or args.debug_max_step is not None:
        ckpt_path = None
    elif model._swa_applied:
        ckpt_path = None
        print("[Node1-1-2-1] Using SWA-averaged weights for test inference.")
    else:
        ckpt_path = "best"
        print("[Node1-1-2-1] SWA not applied (early stopping?). Using best checkpoint.")

    test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)

    # Save test score
    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node1-1-2-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
