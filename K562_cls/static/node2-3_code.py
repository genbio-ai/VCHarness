"""Node 2-3: AIDO.Cell-100M LoRA(r=8) Multi-scale Features + STRING_GNN K=16 2-head Fusion.

Strategy:
- Encode the perturbed gene using AIDO.Cell-100M with LoRA(r=8) on all 18 layers.
- **Multi-scale AIDO feature extraction**: instead of only last-layer summary token,
  concatenate summary tokens from 3 transformer layers (early: layer 6, mid: layer 12,
  late: layer 18 = last). This produces a 640*3=1920-dim multi-resolution transcriptomic
  representation that captures both gene-level and regulatory-level context.
- **STRING_GNN K=16 2-head Neighborhood Attention** (frozen, pre-computed cache):
  aggregates PPI topology context for the perturbed gene via top-16 PPI neighbors
  with 2-head cross-attention (proven +0.04-0.05 F1 over AIDO-only in tree).
- Fusion: concat(AIDO_multiscale [1920], STRING_attn [256]) → [2176]
- Head: Linear(2176→256) → LayerNorm → GELU → Dropout(0.5) → Linear(256→3*6640)
- Loss: Weighted Cross-Entropy + label smoothing ε=0.05
- Optimizer: AdamW (lr=1e-4, wd=2e-2) with 10-epoch linear warmup + CosineAnnealingLR
- EarlyStopping: patience=20, min_delta=1e-3

Key differences from sibling node2-2 (AIDO + STRING K=16 2-head, F1=0.5078):
- Multi-scale AIDO feature (3 layers instead of 1 layer) → richer representation
- Fusion input 2176 instead of 896 → larger information capacity
- Extended patience=20 (vs node2-2's 15) to capture late improvements
- max_epochs=300 to budget for longer training
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
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES    = 6640
N_CLASSES  = 3
AIDO_GENES = 19264      # AIDO.Cell vocabulary size
AIDO_DIR   = "/home/Models/AIDO.Cell-100M"
STRING_DIR = "/home/Models/STRING_GNN"
HIDDEN_DIM = 640        # AIDO.Cell-100M hidden size
STRING_DIM = 256        # STRING_GNN embedding dimension

# Multi-scale layers to extract summary tokens from (0-indexed transformer layers)
# Layer 6 = early (syntactic gene patterns)
# Layer 12 = mid (co-expression context)
# Layer 18 = last (deep regulatory context, last hidden state)
AIDO_LAYERS = [6, 12, 18]   # 3 layers → 3×640 = 1920-dim AIDO feature

CLASS_FREQ = [0.0429, 0.9251, 0.0320]  # class 0=down(-1), class 1=neutral(0), class 2=up(+1) — matches DATA_ABSTRACT freqs

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
    """Per-gene macro-averaged F1 matching calc_metric.py.

    Args:
        preds:   [N, 3, G]   softmax probabilities
        targets: [N, G]      integer class labels in {0, 1, 2}
    Returns:
        scalar F1
    """
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
# STRING GNN Neighborhood Attention Module
# ---------------------------------------------------------------------------
class NeighborhoodAttentionModule(nn.Module):
    """K-nearest neighbor multi-head attention over STRING PPI graph.

    For each perturbed gene, retrieves its top-K PPI neighbors, applies
    multi-head cross-attention (query = center gene, key/value = neighbors),
    and returns a context embedding augmented with STRING edge weight biases.

    Args:
        in_dim:    embedding dimension of STRING_GNN (256)
        n_heads:   number of attention heads
        attn_dim:  dimension per head for Q/K projection
        K:         number of top neighbors to aggregate
    """

    def __init__(
        self,
        in_dim:   int = 256,
        n_heads:  int = 2,
        attn_dim: int = 64,
        K:        int = 16,
    ) -> None:
        super().__init__()
        self.K        = K
        self.n_heads  = n_heads
        self.attn_dim = attn_dim

        self.center_proj   = nn.Linear(in_dim, n_heads * attn_dim, bias=False)
        self.neighbor_proj = nn.Linear(in_dim, n_heads * attn_dim, bias=False)
        self.value_proj    = nn.Linear(in_dim, in_dim, bias=False)
        self.out_proj      = nn.Linear(in_dim, in_dim // 2, bias=False)  # → 128
        self.center_out    = nn.Linear(in_dim, in_dim // 2, bias=False)  # → 128
        self.norm          = nn.LayerNorm(in_dim)

    def forward(
        self,
        center_emb:    torch.Tensor,   # [B, in_dim]
        neighbor_embs: torch.Tensor,   # [B, K, in_dim]
        edge_weights:  torch.Tensor,   # [B, K]  (normalized STRING scores)
    ) -> torch.Tensor:
        """Returns [B, in_dim] = concat(center_out, context_out)."""
        B = center_emb.shape[0]

        # Multi-head cross-attention: query=center, key/value=neighbors
        q = self.center_proj(center_emb).view(B, self.n_heads, self.attn_dim)   # [B, H, A]
        k = self.neighbor_proj(neighbor_embs).view(B, self.K, self.n_heads, self.attn_dim).permute(0, 2, 1, 3)  # [B, H, K, A]
        v = self.value_proj(neighbor_embs).view(B, self.K, self.n_heads, -1).permute(0, 2, 1, 3)               # [B, H, K, D/H]

        # Attention scores + edge weight bias
        scale  = self.attn_dim ** -0.5
        scores = torch.einsum("bha,bhka->bhk", q, k) * scale                   # [B, H, K]
        w_bias = edge_weights.unsqueeze(1).expand_as(scores)                    # [B, H, K]
        scores = scores + w_bias
        attn   = F.softmax(scores, dim=-1)                                       # [B, H, K]

        # Weighted aggregation
        context = torch.einsum("bhk,bhkd->bhd", attn, v).reshape(B, -1)         # [B, D]

        # Project center and context, concatenate
        ctx_out    = self.out_proj(context)                                       # [B, 128]
        center_out = self.center_out(center_emb)                                  # [B, 128]
        return torch.cat([center_out, ctx_out], dim=-1)                           # [B, 256]


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


def make_collate(tokenizer):
    """Factory for collate_fn with AIDO.Cell tokenizer."""

    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]

        # Tokenize: each sample gets only its perturbed gene with expression=1.0
        expr_dicts = [
            {"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids
        ]
        tokenized = tokenizer(expr_dicts, return_tensors="pt")  # input_ids: [B, 19264] float32

        # Find perturbed gene position in the AIDO vocabulary
        input_ids = tokenized["input_ids"]   # [B, 19264]
        gene_in_vocab  = (input_ids > -1.0).any(dim=1)
        gene_positions = torch.where(
            gene_in_vocab,
            (input_ids > -1.0).float().argmax(dim=1),
            torch.zeros(len(batch), dtype=torch.long),
        )   # [B]

        out: Dict[str, Any] = {
            "sample_idx":     torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":        pert_ids,
            "symbol":         symbols,
            "input_ids":      input_ids,
            "attention_mask": tokenized["attention_mask"],
            "gene_positions": gene_positions,
            "gene_in_vocab":  gene_in_vocab,
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

    def setup(self, stage: Optional[str] = None) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_DIR, trust_remote_code=True)

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
            collate_fn=make_collate(self.tokenizer),
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
class AIDOMultiScaleStringFusionModel(pl.LightningModule):
    """AIDO.Cell-100M LoRA(r=8) Multi-scale Features + STRING_GNN K=16 2-head Fusion.

    Architecture:
      AIDO.Cell backbone → extract 3 intermediate summary tokens (layers 6, 12, 18)
        → concat → [B, 1920]
      STRING_GNN K=16 2-head NbAttn → [B, 256]
      concat([B, 1920], [B, 256]) = [B, 2176]
      Linear(2176→256) → LayerNorm → GELU → Dropout(0.5) → Linear(256→3*6640)
    """

    def __init__(
        self,
        lora_r:        int   = 8,
        lora_alpha:    int   = 16,
        lora_dropout:  float = 0.05,
        head_hidden:   int   = 256,
        head_dropout:  float = 0.5,
        lr:            float = 1e-4,
        weight_decay:  float = 2e-2,
        warmup_epochs: int   = 10,
        max_epochs:    int   = 300,
        label_smoothing: float = 0.05,
        nb_K:          int   = 16,
        nb_heads:      int   = 2,
        nb_attn_dim:   int   = 64,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load AIDO.Cell backbone ----
        backbone = AutoModel.from_pretrained(AIDO_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Monkey-patch enable_input_require_grads (AIDO.Cell custom arch)
        backbone.enable_input_require_grads = lambda: None

        # ---- Wrap with LoRA (r=8, all 18 layers) ----
        lora_cfg = LoraConfig(
            task_type      = TaskType.FEATURE_EXTRACTION,
            r              = hp.lora_r,
            lora_alpha     = hp.lora_alpha,
            lora_dropout   = hp.lora_dropout,
            target_modules = ["query", "key", "value"],
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Forward hook to enable grad flow through gene_embedding → LoRA adapters
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        gene_emb = self.backbone.model.bert.gene_embedding
        gene_emb.register_forward_hook(_make_inputs_require_grad)

        # Cast LoRA params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ---- Load STRING_GNN (frozen) ----
        import json as _json
        from transformers import AutoModel as _AM

        string_model = _AM.from_pretrained(STRING_DIR, trust_remote_code=True)
        string_model.eval()
        graph = torch.load(Path(STRING_DIR) / "graph_data.pt", weights_only=False)
        node_names_list = _json.loads((Path(STRING_DIR) / "node_names.json").read_text())

        # Build Ensembl → STRING node index
        self._string_node_map: Dict[str, int] = {
            eid: idx for idx, eid in enumerate(node_names_list)
        }

        # Pre-compute frozen STRING embeddings [18870, 256]
        # Determine the device to run STRING_GNN on
        compute_device = torch.device("cpu")
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            compute_device = torch.device(f"cuda:{local_rank}")
        string_model = string_model.to(compute_device)

        edge_index  = graph["edge_index"].to(compute_device)
        edge_weight = graph["edge_weight"]
        if edge_weight is not None:
            edge_weight = edge_weight.to(compute_device)

        with torch.no_grad():
            string_out = string_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            string_embs = string_out.last_hidden_state.float().cpu()  # [18870, 256]

        # Register as buffer so it moves with the model
        self.register_buffer("string_cache", string_embs)   # [18870, 256]

        # Build adjacency: for each node, store top-K neighbor indices + weights
        # Build neighbor lookup from edge_index and edge_weight
        n_nodes     = string_embs.shape[0]
        K           = hp.nb_K
        edge_index_cpu  = graph["edge_index"].cpu()
        edge_weight_cpu = graph["edge_weight"].cpu() if graph["edge_weight"] is not None else torch.ones(edge_index_cpu.shape[1])

        # Build neighbor lists
        neighbor_idx_list    = [[] for _ in range(n_nodes)]
        neighbor_weight_list = [[] for _ in range(n_nodes)]
        src_arr = edge_index_cpu[0].numpy()
        dst_arr = edge_index_cpu[1].numpy()
        wts_arr = edge_weight_cpu.numpy()
        for s, d, w in zip(src_arr, dst_arr, wts_arr):
            neighbor_idx_list[s].append(d)
            neighbor_weight_list[s].append(float(w))

        # For each node: top-K neighbors by weight
        nbr_idx_mat = torch.zeros(n_nodes, K, dtype=torch.long)
        nbr_wgt_mat = torch.zeros(n_nodes, K, dtype=torch.float32)
        for i in range(n_nodes):
            nbrs = neighbor_idx_list[i]
            wgts = neighbor_weight_list[i]
            if len(nbrs) == 0:
                # isolated node: self-loop
                nbr_idx_mat[i] = i
                nbr_wgt_mat[i] = 0.0
            elif len(nbrs) <= K:
                n = len(nbrs)
                nbr_idx_mat[i, :n] = torch.tensor(nbrs, dtype=torch.long)
                nbr_wgt_mat[i, :n] = torch.tensor(wgts, dtype=torch.float32)
                # Pad remaining with self-loop
                if n < K:
                    nbr_idx_mat[i, n:] = i
                    nbr_wgt_mat[i, n:] = 0.0
            else:
                # Select top-K by weight
                wgt_t = torch.tensor(wgts, dtype=torch.float32)
                nbr_t = torch.tensor(nbrs, dtype=torch.long)
                topk_vals, topk_idx = wgt_t.topk(K)
                nbr_idx_mat[i] = nbr_t[topk_idx]
                nbr_wgt_mat[i] = topk_vals

        self.register_buffer("nbr_idx", nbr_idx_mat)    # [18870, K]
        self.register_buffer("nbr_wgt", nbr_wgt_mat)    # [18870, K]

        # Build pert_id → STRING node index (padded to 0 if not found)
        # We'll compute this per-batch in forward()

        # ---- Neighborhood Attention Module ----
        self.nb_attn = NeighborhoodAttentionModule(
            in_dim   = STRING_DIM,
            n_heads  = hp.nb_heads,
            attn_dim = hp.nb_attn_dim,
            K        = hp.nb_K,
        )

        # ---- Fusion head ----
        # AIDO multi-scale: 3 layers × 640 = 1920
        # STRING neighborhood: 256
        in_dim = len(AIDO_LAYERS) * HIDDEN_DIM + STRING_DIM  # 1920 + 256 = 2176
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

        # Accumulators for validation and test
        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    # ---- forward ----
    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, 19264] float32
        attention_mask: torch.Tensor,   # [B, 19264]
        gene_positions: torch.Tensor,   # [B]  position of perturbed gene in AIDO vocab
        pert_ids:       List[str],       # List[str] Ensembl IDs for STRING lookup
    ) -> torch.Tensor:
        B = input_ids.shape[0]

        # ---- AIDO.Cell multi-scale feature extraction ----
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # Need intermediate layers
        )
        # hidden_states: tuple of 19 tensors [B, 19266, 640] (layer 0 = embedding, 1-18 = transformer)
        # Layer 6 = hidden_states[6], Layer 12 = hidden_states[12], Layer 18 = hidden_states[18]
        hidden_states = out.hidden_states

        aido_parts = []
        for layer_idx in AIDO_LAYERS:
            # Position 19264 is the first summary token (computed via full 18-layer attention)
            # For intermediate layers, use the position-19264 summary token too
            layer_h = hidden_states[layer_idx]   # [B, 19266, 640]
            summary = layer_h[:, AIDO_GENES, :]  # [B, 640]  summary token at pos 19264
            aido_parts.append(summary)

        # Concatenate multi-scale features: [B, 3*640=1920]
        aido_emb = torch.cat(aido_parts, dim=-1).float()   # [B, 1920] cast to fp32

        # ---- STRING_GNN neighborhood attention ----
        # Map pert_ids to STRING node indices
        node_indices = torch.tensor(
            [self._string_node_map.get(pid, 0) for pid in pert_ids],
            dtype=torch.long, device=self.string_cache.device
        )  # [B]

        # Look up center and neighbor embeddings from pre-computed cache
        center_emb  = self.string_cache[node_indices]             # [B, 256]
        nbr_node_idx = self.nbr_idx[node_indices]                  # [B, K]
        nbr_weights  = self.nbr_wgt[node_indices]                  # [B, K]
        neighbor_embs = self.string_cache[nbr_node_idx]            # [B, K, 256]

        string_emb = self.nb_attn(center_emb, neighbor_embs, nbr_weights).float()  # [B, 256]

        # ---- Fusion ----
        fused = torch.cat([aido_emb, string_emb], dim=-1)          # [B, 2176]

        logits = self.head(fused).view(B, N_CLASSES, N_GENES)       # [B, 3, 6640]
        return logits

    # ---- loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),
            targets.reshape(-1),
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    # ---- steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["pert_id"]
        )
        loss = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["pert_id"]
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

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["pert_id"]
        )
        probs = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)
        local_idx   = torch.cat(self._test_idx,   0)
        all_preds   = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_idx     = self.all_gather(local_idx).view(-1)

        if self.trainer.is_global_zero:
            # Sort and deduplicate by sample_idx
            order   = torch.argsort(all_idx)
            s_idx   = all_idx[order]; s_pred = all_preds[order]
            mask    = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
            s_idx   = s_idx[mask]; s_pred = s_pred[mask]

            # Retrieve pert_id/symbol from dataset using integer indices
            test_ds = self.trainer.datamodule.test_ds
            rows = []
            for i in range(s_idx.shape[0]):
                sample_i = s_idx[i].item()
                pid = test_ds.pert_ids[sample_i]
                sym = test_ds.symbols[sample_i]
                rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": json.dumps(s_pred[i].float().cpu().numpy().tolist()),
                })

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            self.print(f"[Node2-3] Saved {len(rows)} test predictions.")

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
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Checkpoint: {trained}/{total} params ({100*trained/total:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- optimizer ----
    def configure_optimizers(self):
        hp = self.hparams
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable_params,
            lr=hp.lr,
            weight_decay=hp.weight_decay,
        )

        # Linear warmup for warmup_epochs, then CosineAnnealingLR
        # We schedule over steps; estimate steps per epoch
        # Use epoch-level scheduler instead for simplicity
        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < hp.warmup_epochs:
                return float(current_epoch + 1) / float(hp.warmup_epochs)
            progress = (current_epoch - hp.warmup_epochs) / max(1, hp.max_epochs - hp.warmup_epochs)
            return max(1e-7 / hp.lr, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "frequency": 1,
            }
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(description="Node2-3: AIDO.Cell Multi-scale + STRING K=16 2-head Fusion")
    parser.add_argument("--micro-batch-size",  type=int,   default=4)
    parser.add_argument("--global-batch-size", type=int,   default=32)
    parser.add_argument("--max-epochs",        type=int,   default=300)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--weight-decay",      type=float, default=2e-2)
    parser.add_argument("--lora-r",            type=int,   default=8)
    parser.add_argument("--lora-alpha",        type=int,   default=16)
    parser.add_argument("--lora-dropout",      type=float, default=0.05)
    parser.add_argument("--head-hidden",       type=int,   default=256)
    parser.add_argument("--head-dropout",      type=float, default=0.5)
    parser.add_argument("--warmup-epochs",     type=int,   default=10)
    parser.add_argument("--label-smoothing",   type=float, default=0.05)
    parser.add_argument("--nb-K",              type=int,   default=16)
    parser.add_argument("--nb-heads",          type=int,   default=2)
    parser.add_argument("--nb-attn-dim",       type=int,   default=64)
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
    model = AIDOMultiScaleStringFusionModel(
        lora_r          = args.lora_r,
        lora_alpha      = args.lora_alpha,
        lora_dropout    = args.lora_dropout,
        head_hidden     = args.head_hidden,
        head_dropout    = args.head_dropout,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        warmup_epochs   = args.warmup_epochs,
        max_epochs      = args.max_epochs,
        label_smoothing = args.label_smoothing,
        nb_K            = args.nb_K,
        nb_heads        = args.nb_heads,
        nb_attn_dim     = args.nb_attn_dim,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
        auto_insert_metric_name = False,
    )
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=20, min_delta=1e-3)
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
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = 1.0,
    )

    trainer.fit(model, datamodule=dm)

    ckpt_path = "best" if (args.debug_max_step is None and not fast_dev_run) else None
    test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)

    score_path = output_dir / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node2-3] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
