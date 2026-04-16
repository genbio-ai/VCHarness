"""Node 1-2 – AIDO.Cell-100M LoRA(r=8) + Frozen STRING_GNN K=16 2-Head Neighborhood Attention.

Strategy:
- AIDO.Cell-100M backbone with LoRA(r=8) fine-tuning on QKV attention projections.
- Encode each perturbation using the AIDO.Cell summary token (position 19264), which
  aggregates global transcriptome state via 18 layers of transformer self-attention.
  This is superior to mean-pooling (99.99% missing gene dilution).
- Augment with frozen STRING_GNN PPI neighborhood attention:
  K=16 top neighbors by STRING edge confidence, 2-head attention (attn_dim=64 each).
  Center-context gating: h = center + alpha * aggregated_neighbors.
- Simple concatenation: [AIDO summary token, 640] + [STRING context, 256] = 896-dim.
- 2-layer MLP head: 896 → 256 → 19920 (3 classes × 6640 genes), dropout=0.5.
- Weighted CE + label smoothing ε=0.05.
- Cosine annealing LR with 10-epoch linear warmup, weight_decay=2e-2.

This mirrors the proven best architecture (node2-1-1-1-1-1, F1=0.5128) but implemented
independently in the node1 lineage. It combines domain-relevant PPI topology from
STRING_GNN with rich transcriptomic representations from AIDO.Cell.

Distinct from sibling node1-1 (pure STRING_GNN bilinear head, F1=0.4527):
- Adds AIDO.Cell-100M backbone with LoRA fine-tuning
- Uses summary token feature (not raw STRING only)
- Dual-stream fusion (AIDO + STRING) vs single-stream STRING
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
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES   = 6640
N_CLASSES = 3
AIDO_GENES = 19264    # AIDO.Cell vocabulary size
AIDO_MODEL_DIR = "/home/Models/AIDO.Cell-100M"
STRING_MODEL_DIR = "/home/Models/STRING_GNN"
AIDO_HIDDEN_DIM = 640      # AIDO.Cell-100M hidden size
STRING_HIDDEN_DIM = 256    # STRING_GNN hidden size

CLASS_FREQ = [0.0429, 0.9251, 0.0320]  # down, neutral, up (remapped 0,1,2)

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency class weights."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py."""
    y_hat = preds.argmax(dim=1)       # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)
        is_pred = (y_hat == c)
        present = is_true.any(dim=0).float()
        tp = (is_pred & is_true).float().sum(0)
        fp = (is_pred & ~is_true).float().sum(0)
        fn = (~is_pred & is_true).float().sum(0)
        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(prec + rec > 0,
                           2 * prec * rec / (prec + rec + 1e-8),
                           torch.zeros_like(prec))
        f1_per_gene += f1_c * present
        n_present   += present

    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# STRING_GNN helpers: pre-compute frozen embeddings + build neighbor index
# ---------------------------------------------------------------------------
def build_string_gnn_resources(device: torch.device) -> Tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
    """
    Returns:
      node_embeddings: [18870, 256] float32 frozen
      pert_id_to_node_idx: dict mapping Ensembl gene ID str -> int (STRING node index)
      neighbor_indices: [18870, K_max] long, padded with -1
      neighbor_weights: [18870, K_max] float32, padded with 0.0
    """
    string_dir = Path(STRING_MODEL_DIR)
    node_names = json.loads((string_dir / "node_names.json").read_text())
    graph = torch.load(string_dir / "graph_data.pt", map_location="cpu")

    # Build pert_id -> node_idx mapping
    pert_id_to_node_idx = {name: idx for idx, name in enumerate(node_names)}

    # Load and run STRING_GNN once to get frozen embeddings
    string_model = AutoModel.from_pretrained(str(string_dir), trust_remote_code=True)
    string_model.eval()
    string_model = string_model.to(device)

    edge_index = graph["edge_index"].to(device)
    edge_weight = graph["edge_weight"]
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    with torch.no_grad():
        outputs = string_model(
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
    node_embeddings = outputs.last_hidden_state.float()  # [18870, 256]

    # Move back to CPU to free GPU memory
    node_embeddings = node_embeddings.cpu()
    del string_model

    # Build adjacency structure for K=16 neighborhood attention
    # Build per-node sorted neighbors (by edge weight descending)
    n_nodes = len(node_names)
    K_max = 16

    # Build adjacency dict: node -> list of (neighbor_idx, weight)
    edge_src = edge_index[0].cpu().numpy()
    edge_dst = edge_index[1].cpu().numpy()
    if edge_weight is not None:
        edge_wts = edge_weight.cpu().numpy()
    else:
        edge_wts = np.ones(len(edge_src), dtype=np.float32)

    adj = [[] for _ in range(n_nodes)]
    for s, d, w in zip(edge_src, edge_dst, edge_wts):
        adj[s].append((d, w))

    # Sort each node's neighbors by weight descending, keep top K_max
    neighbor_indices = torch.full((n_nodes, K_max), -1, dtype=torch.long)
    neighbor_weights = torch.zeros(n_nodes, K_max, dtype=torch.float32)

    for i in range(n_nodes):
        neighbors = sorted(adj[i], key=lambda x: -x[1])[:K_max]
        for j, (nb_idx, nb_wt) in enumerate(neighbors):
            neighbor_indices[i, j] = nb_idx
            neighbor_weights[i, j] = float(nb_wt)  # Convert numpy.float32 to float

    return node_embeddings, pert_id_to_node_idx, neighbor_indices, neighbor_weights


# ---------------------------------------------------------------------------
# 2-Head Neighborhood Attention Module
# ---------------------------------------------------------------------------
class NeighborhoodAttentionModule(nn.Module):
    """
    K=16 PPI neighborhood attention with 2 heads.
    For each perturbed gene, aggregates K=16 top-confidence STRING neighbors
    using 2-head attention (attn_dim=64 per head = 128 total).
    Then gated fusion: h = center + alpha * aggregated.
    """

    def __init__(self, hidden_dim: int = 256, K: int = 16,
                 n_heads: int = 2, attn_dim: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.K = K
        self.n_heads = n_heads
        self.attn_dim = attn_dim

        # Multi-head attention projections
        # For each head: score = W_h(concat(center, neighbor, center-neighbor)) + b_h
        self.attn_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, attn_dim),
                nn.Tanh(),
                nn.Linear(attn_dim, 1, bias=False),
            )
            for _ in range(n_heads)
        ])
        # Combine multi-head attention outputs
        self.head_combine = nn.Linear(hidden_dim * n_heads, hidden_dim)
        # Learnable gating scalar
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        center_emb: torch.Tensor,       # [B, D]
        neighbor_embs: torch.Tensor,    # [B, K, D]
        neighbor_weights: torch.Tensor, # [B, K]
        neighbor_valid: torch.Tensor,   # [B, K] bool mask (True if valid)
    ) -> torch.Tensor:
        B, D = center_emb.shape
        K = neighbor_embs.shape[1]

        # Expand center for concatenation: [B, K, D]
        center_expanded = center_emb.unsqueeze(1).expand_as(neighbor_embs)
        diff = center_expanded - neighbor_embs

        # [B, K, 3D]
        attn_input = torch.cat([center_expanded, neighbor_embs, diff], dim=-1)

        # Compute multi-head attention
        head_outputs = []
        for head in self.attn_heads:
            # [B, K, 1] -> [B, K]
            raw_scores = head(attn_input).squeeze(-1)
            # Add edge weight bias
            raw_scores = raw_scores + neighbor_weights
            # Mask invalid neighbors
            raw_scores = raw_scores.masked_fill(~neighbor_valid, float('-inf'))
            # Softmax over K neighbors
            attn_scores = F.softmax(raw_scores, dim=-1)  # [B, K]
            # Handle all-invalid case (softmax of all -inf gives nan)
            attn_scores = torch.nan_to_num(attn_scores, nan=0.0)
            # Aggregate: [B, K] x [B, K, D] -> [B, D]
            aggregated = (attn_scores.unsqueeze(-1) * neighbor_embs).sum(1)  # [B, D]
            head_outputs.append(aggregated)

        # Combine heads: [B, D*n_heads]
        combined = torch.cat(head_outputs, dim=-1)  # [B, D*n_heads]
        aggregated = self.head_combine(combined)     # [B, D]
        aggregated = self.dropout(aggregated)

        # Gated fusion: center + alpha * aggregated
        alpha = torch.sigmoid(self.alpha)
        output = center_emb + alpha * aggregated
        output = self.layer_norm(output)
        return output


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        has_label = "label" in df.columns and df["label"].notna().all()
        if has_label:
            self.labels: Optional[List] = [
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
            item["labels"] = self.labels[idx]
        return item


def make_collate(tokenizer, pert_id_to_node_idx, neighbor_indices, neighbor_weights):
    """Factory for collate_fn with AIDO.Cell tokenizer and STRING neighbor lookup."""

    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]
        B = len(batch)

        # AIDO.Cell tokenization
        expr_dicts = [
            {"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids
        ]
        tokenized = tokenizer(expr_dicts, return_tensors="pt")
        input_ids = tokenized["input_ids"]  # [B, 19264] float32

        # Find perturbed gene position for AIDO.Cell
        gene_in_vocab  = (input_ids > -1.0).any(dim=1)  # [B]
        gene_positions = torch.where(
            gene_in_vocab,
            (input_ids > -1.0).float().argmax(dim=1),
            torch.zeros(B, dtype=torch.long),
        )

        # STRING_GNN neighbor lookup
        string_node_indices = torch.full((B,), -1, dtype=torch.long)
        for i, pid in enumerate(pert_ids):
            if pid in pert_id_to_node_idx:
                string_node_indices[i] = pert_id_to_node_idx[pid]

        # Gather neighbor indices and weights for batch
        nb_indices = torch.full((B, neighbor_indices.shape[1]), -1, dtype=torch.long)
        nb_weights = torch.zeros(B, neighbor_weights.shape[1], dtype=torch.float32)
        for i, node_idx in enumerate(string_node_indices.tolist()):
            if node_idx >= 0:
                nb_indices[i] = neighbor_indices[node_idx]
                nb_weights[i] = neighbor_weights[node_idx]

        out: Dict[str, Any] = {
            "sample_idx":        torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":           pert_ids,
            "symbol":            symbols,
            "input_ids":         input_ids,
            "attention_mask":    tokenized["attention_mask"],
            "gene_positions":    gene_positions,
            "string_node_indices": string_node_indices,  # [B] long
            "nb_indices":        nb_indices,             # [B, K] long
            "nb_weights":        nb_weights,             # [B, K] float
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out

    return collate_fn


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.tokenizer   = None
        self.pert_id_to_node_idx = None
        self.neighbor_indices = None
        self.neighbor_weights = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Initialize AIDO.Cell tokenizer with barrier
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        # Build STRING_GNN resources (only on rank 0, then broadcast if needed)
        # For simplicity, each rank builds independently (fast since GNN is small)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        (self.node_embeddings,
         self.pert_id_to_node_idx,
         self.neighbor_indices,
         self.neighbor_weights) = build_string_gnn_resources(device)

        # Keep node embeddings and neighbor info on CPU for DataLoader
        # (moved to GPU per-batch in the model)
        self.node_embeddings = self.node_embeddings.cpu()
        self.neighbor_indices = self.neighbor_indices.cpu()
        self.neighbor_weights = self.neighbor_weights.cpu()

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df)
        self.val_ds   = DEGDataset(val_df)
        self.test_ds  = DEGDataset(test_df)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=make_collate(
                self.tokenizer,
                self.pert_id_to_node_idx,
                self.neighbor_indices,
                self.neighbor_weights,
            ),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOStringFusionModel(pl.LightningModule):
    """
    AIDO.Cell-100M LoRA(r=8) + Frozen STRING_GNN K=16 2-head Neighborhood Attention.

    Fusion: concat(AIDO summary token [640], STRING context [256]) → [896]
    Head: Linear(896→256) → LN → GELU → Dropout(0.5) → Linear(256→19920)
    Loss: Weighted CE + label smoothing ε=0.05
    """

    def __init__(
        self,
        lora_r: int         = 8,
        lora_alpha: int     = 16,
        lora_dropout: float = 0.05,
        head_hidden: int    = 256,
        head_dropout: float = 0.5,
        lr: float           = 1e-4,
        weight_decay: float = 2e-2,
        label_smoothing: float = 0.05,
        warmup_epochs: int  = 10,
        max_epochs: int     = 200,
        nb_attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load AIDO.Cell-100M backbone ----
        backbone = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Monkey-patch enable_input_require_grads for AIDO.Cell custom architecture
        backbone.enable_input_require_grads = lambda: None

        # LoRA fine-tuning on QKV attention projections
        lora_cfg = LoraConfig(
            task_type       = TaskType.FEATURE_EXTRACTION,
            r               = hp.lora_r,
            lora_alpha      = hp.lora_alpha,
            lora_dropout    = hp.lora_dropout,
            target_modules  = ["query", "key", "value"],
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Register forward hook to enable gradient flow through gene embedding
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        gene_emb = getattr(self.backbone.model.bert, "gene_embedding", None)
        if gene_emb is not None:
            gene_emb.register_forward_hook(_make_inputs_require_grad)
        else:
            self.print("Warning: backbone.model.bert.gene_embedding not found; skipping hook registration.")

        # Cast LoRA params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ---- STRING_GNN neighborhood attention module ----
        self.nb_attn = NeighborhoodAttentionModule(
            hidden_dim=STRING_HIDDEN_DIM,
            K=16,
            n_heads=2,
            attn_dim=64,
            dropout=hp.nb_attn_dropout,
        )

        # ---- Load frozen STRING_GNN embeddings as buffer ----
        # Access from datamodule (available after setup)
        # We register them as a buffer in on_fit_start/on_test_start
        self._node_emb_registered = False

        # ---- Fusion head ----
        # Input: concat(AIDO summary 640, STRING context 256) = 896
        in_dim = AIDO_HIDDEN_DIM + STRING_HIDDEN_DIM
        self.head = nn.Sequential(
            nn.Linear(in_dim, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )

        # ---- Loss ----
        self.register_buffer("class_weights", get_class_weights())
        self.label_smoothing = hp.label_smoothing

        # Accumulators
        self._val_preds: List[torch.Tensor]   = []
        self._val_tgts:  List[torch.Tensor]   = []
        self._val_idx:   List[torch.Tensor]   = []
        self._test_preds: List[torch.Tensor]  = []
        self._test_idx:  List[torch.Tensor]   = []

    def _register_string_emb(self):
        """Register STRING node embeddings from datamodule as buffer on the correct device."""
        if not self._node_emb_registered and hasattr(self, 'trainer') and self.trainer is not None:
            dm = self.trainer.datamodule
            if dm is not None and hasattr(dm, 'node_embeddings') and dm.node_embeddings is not None:
                node_emb = dm.node_embeddings.float().to(self.device)
                # Register as a non-persistent buffer (not saved in checkpoint)
                self.register_buffer("string_node_emb", node_emb, persistent=False)
                self._node_emb_registered = True

    def on_fit_start(self) -> None:
        self._register_string_emb()

    def on_test_start(self) -> None:
        self._register_string_emb()

    def on_validation_start(self) -> None:
        self._register_string_emb()

    def _get_string_context(
        self,
        string_node_indices: torch.Tensor,  # [B] long
        nb_indices: torch.Tensor,           # [B, K] long
        nb_weights: torch.Tensor,           # [B, K] float
    ) -> torch.Tensor:
        """Get STRING context embedding via neighborhood attention."""
        B = string_node_indices.shape[0]
        device = string_node_indices.device

        # Ensure string_node_emb is available
        if not hasattr(self, 'string_node_emb'):
            # Fallback: return zeros (shouldn't happen in normal flow)
            return torch.zeros(B, STRING_HIDDEN_DIM, device=device, dtype=torch.float32)

        node_emb = self.string_node_emb.to(device)  # [N, 256], ensure on correct device

        # Get center embeddings
        # For genes not in STRING, use zero embedding
        in_string = (string_node_indices >= 0)  # [B]
        safe_indices = string_node_indices.clamp(min=0)  # avoid -1 index
        center_emb = node_emb[safe_indices]  # [B, 256]
        center_emb = center_emb * in_string.float().unsqueeze(-1)  # zero for unknown

        # Get neighbor embeddings
        K = nb_indices.shape[1]
        valid_nb = (nb_indices >= 0)  # [B, K] bool
        safe_nb_indices = nb_indices.clamp(min=0)
        # [B, K, 256]
        neighbor_embs = node_emb[safe_nb_indices.view(-1)].view(B, K, STRING_HIDDEN_DIM)
        # Zero out invalid neighbors
        neighbor_embs = neighbor_embs * valid_nb.float().unsqueeze(-1)

        # Apply neighborhood attention
        string_context = self.nb_attn(
            center_emb=center_emb.float(),
            neighbor_embs=neighbor_embs.float(),
            neighbor_weights=nb_weights.float(),
            neighbor_valid=valid_nb,
        )  # [B, 256]

        return string_context

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_positions: torch.Tensor,
        string_node_indices: torch.Tensor,
        nb_indices: torch.Tensor,
        nb_weights: torch.Tensor,
    ) -> torch.Tensor:
        B = input_ids.shape[0]

        # --- AIDO.Cell forward ---
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        # last_hidden_state: [B, 19266, 640]
        lhs = out.last_hidden_state

        # Summary token at position 19264 (first summary token)
        summary_emb = lhs[:, AIDO_GENES, :].float()  # [B, 640]

        # --- STRING_GNN context ---
        string_context = self._get_string_context(
            string_node_indices, nb_indices, nb_weights
        )  # [B, 256]

        # --- Fusion: simple concatenation ---
        fused = torch.cat([summary_emb, string_context], dim=-1)  # [B, 896]

        # --- Head ---
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)  # [B, 3, G]
        return logits

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),
            targets.reshape(-1),
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["input_ids"], batch["attention_mask"], batch["gene_positions"],
            batch["string_node_indices"], batch["nb_indices"], batch["nb_weights"],
        )
        loss = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"], batch["gene_positions"],
            batch["string_node_indices"], batch["nb_indices"], batch["nb_weights"],
        )
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits.float(), dim=1).detach()
            self._val_preds.append(probs.cpu())
            self._val_tgts.append(batch["labels"].detach().cpu())
            self._val_idx.append(batch["sample_idx"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, 0)
        local_tgts  = torch.cat(self._val_tgts,  0)
        local_idx   = torch.cat(self._val_idx,   0)
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        # Move to device for all_gather
        device = self.device
        local_preds = local_preds.to(device)
        local_tgts  = local_tgts.to(device)
        local_idx   = local_idx.to(device)

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order];  s_pred = all_preds[order];  s_tgt = all_tgts[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)
        self.log("val_f1", f1, prog_bar=False, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"], batch["gene_positions"],
            batch["string_node_indices"], batch["nb_indices"], batch["nb_weights"],
        )
        probs = torch.softmax(logits.float(), dim=1).detach().cpu()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach().cpu())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)   # [local_N, 3, G]
        local_idx   = torch.cat(self._test_idx,   0)   # [local_N]

        # Move to device for all_gather
        local_preds = local_preds.to(self.device)
        local_idx   = local_idx.to(self.device)

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        if self.trainer.is_global_zero:
            # Deduplicate by sample_idx
            order  = torch.argsort(all_idx)
            s_idx  = all_idx[order]
            s_pred = all_preds[order]
            mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
            unique_idx  = s_idx[mask].cpu()
            unique_pred = s_pred[mask].cpu()

            # Get pert_id and symbol from test dataset
            test_ds = self.trainer.datamodule.test_ds
            rows = []
            for j in range(len(unique_idx)):
                ds_idx = unique_idx[j].item()
                pid = test_ds.pert_ids[ds_idx]
                sym = test_ds.symbols[ds_idx]
                rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": json.dumps(unique_pred[j].float().numpy().tolist()),
                })

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1-2] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()

    # ---- checkpoint ----
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
        total  = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Checkpoint: {trained}/{total} params ({100*trained/total:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- optimizer + scheduler ----
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # Linear warmup then cosine annealing
        max_epochs = self.hparams.max_epochs
        warmup_epochs = self.hparams.warmup_epochs

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            return max(1e-7 / self.hparams.lr, 0.5 * (1 + np.cos(np.pi * progress)))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "epoch",
                "frequency": 1,
            }
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node1-2 – AIDO.Cell-100M LoRA + STRING_GNN K=16 2-Head Neighborhood Attention"
    )
    parser.add_argument("--micro-batch-size",  type=int,   default=4)
    parser.add_argument("--global-batch-size", type=int,   default=32)
    parser.add_argument("--max-epochs",        type=int,   default=200)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--weight-decay",      type=float, default=2e-2)
    parser.add_argument("--lora-r",            type=int,   default=8)
    parser.add_argument("--lora-alpha",        type=int,   default=16)
    parser.add_argument("--lora-dropout",      type=float, default=0.05)
    parser.add_argument("--head-hidden",       type=int,   default=256)
    parser.add_argument("--head-dropout",      type=float, default=0.5)
    parser.add_argument("--label-smoothing",   type=float, default=0.05)
    parser.add_argument("--warmup-epochs",     type=int,   default=10)
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--debug-max-step",    type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",      action="store_true", dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = args.debug_max_step
        lim_val   = args.debug_max_step
        lim_test  = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = lim_val = lim_test = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDOStringFusionModel(
        lora_r          = args.lora_r,
        lora_alpha      = args.lora_alpha,
        lora_dropout    = args.lora_dropout,
        head_hidden     = args.head_hidden,
        head_dropout    = args.head_dropout,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        label_smoothing = args.label_smoothing,
        warmup_epochs   = args.warmup_epochs,
        max_epochs      = args.max_epochs,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val_f1:.4f}",
        monitor    = "val_f1",
        mode       = "max",
        save_top_k = 1,
        auto_insert_metric_name = False,
    )
    es_cb = EarlyStopping(monitor="val_f1", mode="max", patience=15, min_delta=1e-3)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

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
        val_check_interval      = 1.0 if (args.debug_max_step is not None or fast_dev_run) else args.val_check_interval,
        num_sanity_val_steps    = 2,
        callbacks               = [ckpt_cb, es_cb, lr_cb, pg_cb],
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 5,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = 1.0,
    )

    trainer.fit(model, datamodule=dm)

    ckpt_path = "best" if (args.debug_max_step is None and not fast_dev_run) else None
    test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)

    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node1-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
