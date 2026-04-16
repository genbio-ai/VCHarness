"""Node 3-1-3-1: AIDO.Cell-100M + Last-6-Layer QKV Muon + Frozen STRING_GNN Direct Lookup.

Improvements from parent node3-1-3 (test F1=0.188 catastrophic failure):

PRIMARY CHANGE: Scale AIDO.Cell backbone from 10M to 100M.
  - Parent feedback explicitly recommends: "Scale up to AIDO.Cell-100M rather than
    further engineering of STRING features."
  - node2-1-1-1 (AIDO.Cell-100M + LoRA + STRING) achieved F1=0.5059 (+0.31 over parent)
  - AIDO.Cell-100M: 640-dim hidden, 18 transformer layers (vs 256-dim, 8 layers for 10M)

SECONDARY CHANGE: Remove NeighborhoodAttentionGated module.
  - The learned K=16 neighborhood attention on frozen STRING embeddings in the AIDO
    hybrid context caused IDENTICAL catastrophic failure twice:
      * Parent node3-1 (focal loss + neighborhood attn): F1=0.188
      * Parent node3-1-3 (label-smooth CE + neighborhood attn): F1=0.188
  - Root cause: frozen STRING embeddings + learned attention = fragile features that
    fail to generalize beyond the 1388-sample training set
  - Sibling node3-1-2 (direct STRING lookup, NO neighborhood attn): F1=0.4407 (stable)

TERTIARY CHANGE: Direct STRING lookup (no learned transformation).
  - Sibling node3-1-2 proves direct lookup is stable (val F1 = test F1 = 0.4407)
  - Avoids the fragile learned attention that causes test generalization collapse

QUATERNARY CHANGE: Last 6 layers QKV fine-tuning (instead of all 8 layers for 10M).
  - AIDO.Cell-100M has 18 layers; fine-tuning all QKV gives 22M trainable params
    (vs 1.6M for 10M), which risks overfitting on 1388 samples
  - Last 6 layers QKV: 640×640×3×6 = 7.4M trainable params (conservative)
  - Analogous to "top-6 layers" strategy proven in scFoundation lineage (node4-2: F1=0.4801)

Architecture:
- AIDO.Cell-100M (640-dim): last 6 layers' QKV weights fine-tuned with Muon
  * FlashAttention enabled (required for 19266-length sequences)
  * Weight sharing: flash_self.{q,k,v}.weight = self.{q,k,v}.weight (shared tensor)
  * This ensures FlashAttention uses the trained weights
- STRING_GNN: frozen, direct embedding lookup (pert_id → 256-dim)
  * No learned transformation, no generalization risk
- Fusion: simple concat [AIDO_640 + STRING_256] = 896-dim
- Head: 896 → 256 (LN + GELU + Dropout=0.4) → 19920 → reshape [B, 3, 6640]
- Loss: label-smoothed CE (ε=0.1) + sqrt-inverse-freq class weights
- Optimizer: Muon for 640×640 QKV weight matrices, AdamW for head
- Schedule: CosineAnnealingLR (T_max=120, no warm restarts)
- Gradient checkpointing: enabled for memory efficiency

Expected F1: 0.45-0.50+ (between sibling node3-1-2 at 0.4407 and node2-1-1-1 at 0.5059)
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES          = 6640
N_CLASSES        = 3
AIDO_GENES       = 19264
AIDO_MODEL_DIR   = "/home/Models/AIDO.Cell-100M"
STRING_GNN_DIR   = "/home/Models/STRING_GNN"
HIDDEN_DIM_100M  = 640   # AIDO.Cell-100M hidden dimension
STRING_DIM       = 256   # STRING_GNN output dimension
STRING_HIDDEN_DIM = 256  # STRING_GNN output dimension (alias)
N_AIDO_LAYERS    = 18    # AIDO.Cell-100M transformer depth
HEAD_IN_DIM      = HIDDEN_DIM_100M + STRING_DIM  # 896

# Class frequencies in training labels (down-regulated, neutral, up-regulated)
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency class weights for 92.5% neutral class imbalance."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1, exactly matching data/calc_metric.py logic.

    Args:
        preds:   [N, 3, G] softmax probabilities
        targets: [N, G]    integer class labels in {0,1,2}
    Returns:
        scalar F1 averaged over all G genes
    """
    y_hat       = preds.argmax(dim=1)  # [N, G]
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
# Dataset / DataModule
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


def make_collate(tokenizer):
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]

        # Set expression=1.0 for the perturbed gene only; all others default to -1.0 (missing)
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        input_ids  = tokenized["input_ids"]  # [B, 19264] float32

        # Find the position of the perturbed gene in the fixed 19264-gene sequence
        gene_in_vocab  = (input_ids > -1.0).any(dim=1)    # [B] bool
        gene_positions = torch.where(
            gene_in_vocab,
            (input_ids > -1.0).float().argmax(dim=1),     # position of perturbed gene
            torch.zeros(len(batch), dtype=torch.long),    # fallback: position 0
        )

        out: Dict[str, Any] = {
            "sample_idx":     torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":        pert_ids,
            "symbol":         symbols,
            "input_ids":      input_ids,
            "attention_mask": tokenized["attention_mask"],
            "gene_positions": gene_positions,
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out
    return collate_fn


class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.tokenizer   = None

    def setup(self, stage: Optional[str] = None) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        self.train_ds = DEGDataset(pd.read_csv(TRAIN_TSV, sep="\t"))
        self.val_ds   = DEGDataset(pd.read_csv(VAL_TSV,   sep="\t"))
        self.test_ds  = DEGDataset(pd.read_csv(TEST_TSV,  sep="\t"))

    def _loader(self, ds: DEGDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=make_collate(self.tokenizer),
        )

    def train_dataloader(self) -> DataLoader: return self._loader(self.train_ds, True)
    def val_dataloader(self)   -> DataLoader: return self._loader(self.val_ds,   False)
    def test_dataloader(self)  -> DataLoader: return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------
class AIDO100MStringDirectModel(pl.LightningModule):
    """AIDO.Cell-100M (last-N-layer QKV Muon) + frozen STRING_GNN direct lookup.

    Architecture:
      AIDO.Cell-100M ──(last 6 layers' QKV, Muon)──> gene_pos hidden [B, 640]
                                                               |
      STRING_GNN (frozen) ──(direct lookup by pert_id)──> [B, 256]
                                                               |
                           concat ──────────────────────> [B, 896]
                                                               |
                           MLP head ─────────────────────> [B, 3, 6640]

    Design rationale:
    - Scale-up to 100M: primary recommendation from parent failure analysis
    - Direct STRING lookup: avoids the learned attention fragility (caused F1=0.188 twice)
    - Last 6 layers QKV: balances capacity (7.4M params) vs overfitting risk (1388 samples)
    - Muon: optimal for 640×640 square QKV matrices (larger than 256×256 in 10M)
    """

    def __init__(
        self,
        fine_tune_last_n:  int   = 6,      # number of last layers to fine-tune QKV
        head_hidden:       int   = 256,    # classification head hidden dimension
        head_dropout:      float = 0.4,    # head dropout rate
        lr_muon:           float = 0.02,   # Muon LR for QKV weight matrices
        lr_adamw:          float = 2e-4,   # AdamW LR for classification head
        weight_decay:      float = 2e-2,   # L2 regularization
        max_epochs:        int   = 120,    # CosineAnnealingLR T_max
        label_smoothing:   float = 0.1,    # label smoothing epsilon
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load AIDO.Cell-100M backbone ----
        self.backbone = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        self.backbone = self.backbone.to(torch.bfloat16)
        self.backbone.config.use_cache = False

        # Gradient checkpointing: reduces activation memory from ~9 GiB/sample to ~3.5 GiB
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Enable FlashAttention-2 to avoid O(n^2) memory for 19266-length sequences
        # (Without FA, attention matrix is [B, 20 heads, 19266, 19266] ≈ 57 GB per batch!)
        self.backbone.config._use_flash_attention_2 = True

        # CRITICAL: Share QKV weight tensors between flash_self and regular self paths.
        # Since FlashAttention uses the flash_self path, we must ensure the weights are
        # shared (same Python tensor object) so that training flash_self weights = training
        # self weights. After sharing, setting requires_grad=True on one affects both.
        # Note: AIDO.Cell-100M has add_linear_bias=False, so no bias sharing needed.
        for layer in self.backbone.bert.encoder.layer:
            ss = layer.attention.flash_self   # BertSelfFlashAttention
            mm = layer.attention.self          # CellFoundationSelfAttention
            ss.query.weight = mm.query.weight
            ss.key.weight   = mm.key.weight
            ss.value.weight = mm.value.weight

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze only the last fine_tune_last_n layers' QKV weight matrices
        # (last 6 of 18 layers → 640×640×3×6 = 7.37M trainable params)
        total_layers = len(self.backbone.bert.encoder.layer)  # 18 for AIDO.Cell-100M
        qkv_suffixes = (
            "attention.self.query.weight",
            "attention.self.key.weight",
            "attention.self.value.weight",
        )
        for name, param in self.backbone.named_parameters():
            if not any(name.endswith(s) for s in qkv_suffixes):
                continue
            # Name format: "bert.encoder.layer.{idx}.attention.self.{q/k/v}.weight"
            parts = name.split(".")
            try:
                layer_pos = parts.index("layer")
                layer_idx = int(parts[layer_pos + 1])
                if layer_idx >= total_layers - hp.fine_tune_last_n:
                    param.requires_grad = True
            except (ValueError, IndexError):
                pass

        # Keep trainable QKV in bf16 (required for FlashAttention fast path compatibility)
        # Note: Muon handles bf16 parameters correctly via its Newton-Schulz update step
        qkv_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_backbone = sum(p.numel() for p in self.backbone.parameters())
        print(
            f"[Node3-1-3-1] AIDO.Cell-100M trainable QKV (last {hp.fine_tune_last_n} layers): "
            f"{qkv_trainable:,}/{total_backbone:,} params"
        )

        # ---- Load STRING_GNN (completely frozen) ----
        self.string_gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
        for param in self.string_gnn.parameters():
            param.requires_grad = False
        self.string_gnn.eval()

        # Load and register STRING graph tensors as buffers (auto-moved to GPU with model)
        graph = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt")
        self.register_buffer("edge_index", graph["edge_index"])
        ew = graph.get("edge_weight", None)
        if ew is not None:
            self.register_buffer("edge_weight", ew.float())
        else:
            self.register_buffer("edge_weight", None)

        # Build Ensembl gene ID → STRING node index mapping
        node_names = json.loads((Path(STRING_GNN_DIR) / "node_names.json").read_text())
        self.string_name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(node_names)}
        print(f"[Node3-1-3-1] STRING_GNN loaded: {len(node_names)} nodes")

        # ---- Classification head ----
        # Input: AIDO.Cell-100M gene embedding (640) + STRING direct lookup (256) = 896
        in_dim = HEAD_IN_DIM  # 896
        self.head = nn.Sequential(
            nn.Linear(in_dim, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),  # 3 × 6640 = 19920
        )
        # Cast head to float32 for stable gradient flow (critical for small heads on bf16 backbone)
        for param in self.head.parameters():
            param.data = param.data.float()

        # Loss: label-smoothed CE + class weights
        self.register_buffer("class_weights", get_class_weights())

        # State buffers for epoch-level metric aggregation (DDP-safe)
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    # ---- Forward pass ----
    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, 19264] float32
        attention_mask: torch.Tensor,   # [B, 19264] int64
        gene_positions: torch.Tensor,   # [B] int64 — position of perturbed gene in sequence
        pert_ids:       List[str],       # Ensembl gene IDs
    ) -> torch.Tensor:
        """
        Returns:
            logits: [B, N_CLASSES, N_GENES] = [B, 3, 6640]
        """
        B      = input_ids.shape[0]
        device = input_ids.device

        # ---- AIDO.Cell-100M forward ----
        # Output: last_hidden_state [B, 19266, 640] (19264 genes + 2 summary tokens appended)
        aido_out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        # Extract the embedding at the perturbed gene's position in the sequence
        # gene_positions ∈ {0, ..., 19263} → valid gene positions (not summary tokens)
        aido_feat = aido_out.last_hidden_state[
            torch.arange(B, device=device), gene_positions, :
        ].float()  # [B, 640]

        # ---- STRING_GNN: frozen direct embedding lookup ----
        with torch.no_grad():
            string_all_embs = self.string_gnn(
                edge_index=self.edge_index,
                edge_weight=self.edge_weight,
            ).last_hidden_state.float()  # [N_STRING_NODES, 256]

        # Map Ensembl pert_ids to STRING node indices
        # Genes not in STRING graph get index -1 → clamped to 0, then zeroed out
        gene_idx_list   = [self.string_name_to_idx.get(pid, -1) for pid in pert_ids]
        gene_idx_tensor = torch.tensor(gene_idx_list, dtype=torch.long, device=device)
        valid_mask      = (gene_idx_tensor >= 0).float().unsqueeze(1)   # [B, 1]
        clamped_idx     = gene_idx_tensor.clamp(min=0)                  # [B]
        string_feat     = string_all_embs[clamped_idx] * valid_mask     # [B, 256]

        # ---- Fusion: simple concatenation ----
        # AIDO (640) + STRING direct (256) = 896-dim
        fused  = torch.cat([aido_feat, string_feat], dim=-1)   # [B, 896]
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)  # [B, 3, 6640]
        return logits

    # ---- Loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G   = logits.shape
        flat_log  = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, C]
        flat_tgt  = targets.reshape(-1)                      # [B*G]
        return F.cross_entropy(
            flat_log, flat_tgt,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ---- Lightning steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["pert_id"],
        )
        loss = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["pert_id"],
        )
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

        # Gather from all DDP ranks and deduplicate by sample index
        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        dedup  = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
        f1 = compute_per_gene_f1(s_pred[dedup], s_tgt[dedup])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["pert_id"],
        )
        probs = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)
        local_idx   = torch.cat(self._test_idx,   0)
        self._test_preds.clear(); self._test_idx.clear()

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        if self.trainer.is_global_zero:
            order   = torch.argsort(all_idx)
            s_idx   = all_idx[order]
            s_pred  = all_preds[order]
            dedup   = torch.cat([
                torch.tensor([True], device=s_idx.device),
                s_idx[1:] != s_idx[:-1],
            ])
            s_idx  = s_idx[dedup]
            s_pred = s_pred[dedup]

            test_ds = self.trainer.datamodule.test_ds
            rows = []
            for i, idx in enumerate(s_idx.cpu().tolist()):
                pid = test_ds.pert_ids[idx]
                sym = test_ds.symbols[idx]
                rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": json.dumps(s_pred[i].float().cpu().numpy().tolist()),
                })

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node3-1-3-1] Saved {len(rows)} test predictions.")

    # ---- Checkpoint: save only trainable parameters + buffers ----
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
            f"Checkpoint: {trained:,}/{total:,} params ({100 * trained / total:.2f}%)"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer: Muon for QKV, AdamW for head ----
    def configure_optimizers(self):
        hp = self.hparams

        # Muon group: AIDO.Cell-100M QKV weight matrices of last N layers (640×640, square)
        # These tensors are shared between flash_self and self paths via weight sharing above.
        # Using p.ndim >= 2 to select weight matrices (not biases/scalars, though none exist).
        qkv_weights = [
            p for p in self.backbone.parameters()
            if p.requires_grad and p.ndim >= 2
        ]

        # AdamW group: classification head parameters
        adamw_params = list(self.head.parameters())

        if not qkv_weights:
            raise ValueError("No trainable QKV weight matrices found in backbone!")
        if not adamw_params:
            raise ValueError("No head parameters found!")

        print(
            f"[Node3-1-3-1] Muon params: {sum(p.numel() for p in qkv_weights):,} | "
            f"AdamW params: {sum(p.numel() for p in adamw_params):,}"
        )

        param_groups = [
            dict(
                params       = qkv_weights,
                use_muon     = True,
                lr           = hp.lr_muon,
                weight_decay = hp.weight_decay,
                momentum     = 0.95,
            ),
            dict(
                params       = adamw_params,
                use_muon     = False,
                lr           = hp.lr_adamw,
                betas        = (0.9, 0.95),
                weight_decay = hp.weight_decay,
            ),
        ]

        use_distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        opt_cls   = MuonWithAuxAdam if use_distributed else SingleDeviceMuonWithAuxAdam
        optimizer = opt_cls(param_groups)

        # CosineAnnealingLR: smooth monotonic decay, no warm restarts
        # Avoids the SGDR restart disruption observed in sibling node3-1-2
        # (which caused val F1 to drop from 0.441 → 0.433 at epoch 15)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hp.max_epochs, eta_min=1e-7
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
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
        description="Node3-1-3-1: AIDO.Cell-100M + Last-6-Layer QKV Muon + Frozen STRING Direct"
    )
    parser.add_argument("--micro-batch-size",    type=int,   default=8)
    parser.add_argument("--global-batch-size",   type=int,   default=128)
    parser.add_argument("--max-epochs",          type=int,   default=120)
    parser.add_argument("--lr-muon",             type=float, default=0.02)
    parser.add_argument("--lr-adamw",            type=float, default=2e-4)
    parser.add_argument("--weight-decay",        type=float, default=2e-2)
    parser.add_argument("--fine-tune-last-n",    type=int,   default=6)
    parser.add_argument("--head-hidden",         type=int,   default=256)
    parser.add_argument("--head-dropout",        type=float, default=0.4)
    parser.add_argument("--label-smoothing",     type=float, default=0.1)
    parser.add_argument("--num-workers",         type=int,   default=4)
    parser.add_argument("--debug_max_step",      type=int,   default=None)
    parser.add_argument("--fast_dev_run",        action="store_true")
    parser.add_argument("--val-check-interval",  type=float, default=1.0)
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = lim_val = lim_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = lim_val = lim_test = 1.0
        max_steps = -1

    # Gradient accumulation to reach global batch size
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    print(
        f"[Node3-1-3-1] n_gpus={n_gpus}, micro_bs={args.micro_batch_size}, "
        f"global_bs={args.global_batch_size}, accumulate={accumulate}"
    )

    # Model and data
    model = AIDO100MStringDirectModel(
        fine_tune_last_n = args.fine_tune_last_n,
        head_hidden      = args.head_hidden,
        head_dropout     = args.head_dropout,
        lr_muon          = args.lr_muon,
        lr_adamw         = args.lr_adamw,
        weight_decay     = args.weight_decay,
        max_epochs       = args.max_epochs,
        label_smoothing  = args.label_smoothing,
    )
    datamodule = DEGDataModule(
        batch_size  = args.micro_batch_size,
        num_workers = args.num_workers,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath        = str(output_dir / "checkpoints"),
        filename       = "best-{epoch:03d}-{val/f1:.4f}",
        monitor        = "val/f1",
        mode           = "max",
        save_top_k     = 1,
        save_last      = True,
        auto_insert_metric_name = False,
    )
    early_stop_callback = EarlyStopping(
        monitor    = "val/f1",
        mode       = "max",
        patience   = 12,        # Enough to capture late-stage improvements
        min_delta  = 0.002,     # Ignore noise oscillations < 0.2%
        verbose    = True,
    )
    lr_monitor    = LearningRateMonitor(logging_interval="epoch")
    progress_bar  = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger         = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tensorboard_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy: DDP with find_unused_parameters for safety
    strategy = DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=180))

    trainer = pl.Trainer(
        accelerator             = "gpu",
        devices                 = n_gpus,
        num_nodes               = 1,
        strategy                = strategy,
        precision               = "bf16-mixed",
        max_epochs              = args.max_epochs,
        max_steps               = max_steps,
        accumulate_grad_batches = accumulate,
        limit_train_batches     = lim_train,
        limit_val_batches       = lim_val,
        limit_test_batches      = lim_test,
        val_check_interval      = (
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps    = 2,
        callbacks               = [checkpoint_callback, early_stop_callback, lr_monitor, progress_bar],
        logger                  = [csv_logger, tensorboard_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = 1.0,   # Gradient clipping for stable training
    )

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Test on best checkpoint
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Save test score
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_val  = test_results[0].get("test/loss", float("nan"))
        # Note: test score (F1) is computed by EvaluateAgent from test_predictions.tsv
        score_path.write_text(f"test_loss={score_val:.6f}\n")
        print(f"[Node3-1-3-1] Test results: {test_results}")


if __name__ == "__main__":
    main()
