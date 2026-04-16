"""Node 2-3-1: AIDO.Cell-100M LoRA(r=8) + Cross-Layer Attention Fusion (layers 16,17,18) + STRING_GNN K=16 2-head.

Strategy:
- This node directly addresses the parent node2-3's failure: raw concatenation of summary tokens
  from layers [6, 12, 18] → 1920-dim input overwhelmed the head and introduced noise,
  resulting in a severe -0.061 F1 regression (0.4473 vs 0.5078 baseline).
- Primary innovation (feedback-recommended): Replace raw concatenation with
  **cross-layer attention fusion** over near-final layers [16, 17, 18]:
  - Query: last-layer (18) summary token
  - Keys/Values: summary tokens from layers [16, 17, 18]
  - Output: 640-dim attention-weighted representation (same dim as proven single-layer best!)
  - Added params: ~82K (2 × 640×64 linear projections)
  - Near-final layers (16, 17, 18) all have full perturbation context (unlike shallow layers 6 or 12)
  - Attention can learn to upweight/downweight individual layers rather than forcing all layers
- Architecture: concat(AIDO_attn [640], STRING_K16_2head [256]) = 896-dim → 256-dim head
  (identical fusion dim to tree best node2-1-1-1-1-1, F1=0.5128)
- All proven regularization retained: LoRA r=8, dropout=0.5, weighted CE + LS(0.05),
  cosine annealing (10-ep warmup), AdamW (lr=1e-4, wd=2e-2), patience=15

Key differences from parent (node2-3):
  - REMOVE raw concat [6,12,18] → 1920-dim
  - ADD cross-layer attention [16,17,18] → 640-dim (attention-pooled, not concatenated)
  - Fusion input: 2176-dim → 896-dim (matches proven best!)
  - max_epochs: 300 → 250

Key differences from sibling node2-2 (AIDO single-layer, F1=0.5078):
  - Uses attention fusion of layers [16,17,18] instead of just layer 18
  - Near-final layers can provide complementary refinement that single-layer misses
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

# Near-final transformer layers for cross-layer attention fusion
# Layers 16, 17, 18 all have full perturbation context (unlike shallow layers 6-12)
# Using only near-final layers avoids the noise that destroyed the parent node's performance
FUSION_LAYERS = [16, 17, 18]   # 3 near-final layers for attention fusion → 640-dim out
LAYER_ATTN_PROJ_DIM = 64       # Q/K projection dim for layer attention

CLASS_FREQ = [0.0429, 0.9251, 0.0320]  # class 0=down(-1), class 1=neutral(0), class 2=up(+1)

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
# Cross-Layer Attention Fusion Module
# ---------------------------------------------------------------------------
class LayerAttentionFusion(nn.Module):
    """Cross-layer attention fusion for near-final AIDO.Cell transformer layers.

    Takes summary tokens from multiple near-final transformer layers and produces
    a single hidden_dim representation via learned cross-attention.

    The last layer's summary token is used as the query; all layers' summary tokens
    serve as keys and values. This allows the model to selectively incorporate
    information from earlier near-final layers while keeping layer-18 as the
    primary representation anchor.

    Only adds ~2 × hidden_dim × proj_dim trainable parameters (~82K for hidden=640, proj=64).
    Output dimension is identical to the single-layer approach (hidden_dim=640),
    ensuring the downstream fusion head has the same capacity as the proven best.

    Args:
        hidden_dim: AIDO.Cell hidden size (640 for 100M)
        n_layers:   number of layers being fused (default: 3 for [16, 17, 18])
        proj_dim:   projection dimension for Q/K attention scores
    """

    def __init__(
        self,
        hidden_dim: int = 640,
        n_layers:   int = 3,
        proj_dim:   int = 64,
    ) -> None:
        super().__init__()
        self.proj_q = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.proj_k = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.scale  = proj_dim ** -0.5

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, n_layers, hidden_dim]  - summary tokens from multiple layers
        Returns:
            [B, hidden_dim]  - attention-weighted combination
        """
        B, L, D = tokens.shape
        # Use last-layer token (tokens[:, -1, :]) as query (= layer 18 = fully refined)
        q = self.proj_q(tokens[:, -1, :])  # [B, proj_dim]
        k = self.proj_k(tokens)             # [B, n_layers, proj_dim]

        # Attention scores: [B, n_layers]
        scores = torch.einsum('bd,bld->bl', q, k) * self.scale
        attn   = F.softmax(scores, dim=-1)  # [B, n_layers]

        # Attention-weighted sum of original (un-projected) tokens
        out = (attn.unsqueeze(-1) * tokens).sum(dim=1)  # [B, hidden_dim]
        return out


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
class AIDOLayerAttentionStringFusionModel(pl.LightningModule):
    """AIDO.Cell-100M LoRA(r=8) + Cross-Layer Attention Fusion (layers 16,17,18) + STRING_GNN K=16 2-head.

    Architecture:
      AIDO.Cell backbone → extract summary tokens from near-final layers [16, 17, 18]
        → LayerAttentionFusion (cross-attn, query=layer18) → [B, 640]
      STRING_GNN K=16 2-head NbAttn → [B, 256]
      concat([B, 640], [B, 256]) = [B, 896]   ← same dim as proven best node2-1-1-1-1-1 (F1=0.5128)
      Linear(896→256) → LayerNorm → GELU → Dropout(0.5) → Linear(256→3*6640)

    Key change from parent (node2-3):
      Replacing raw concatenation of 3 layers → 1920-dim (failed, F1=0.4473) with
      attention-pooled fusion → 640-dim (matches proven best architecture).
    """

    def __init__(
        self,
        lora_r:          int   = 8,
        lora_alpha:      int   = 16,
        lora_dropout:    float = 0.05,
        head_hidden:     int   = 256,
        head_dropout:    float = 0.5,
        lr:              float = 1e-4,
        weight_decay:    float = 2e-2,
        warmup_epochs:   int   = 10,
        max_epochs:      int   = 250,
        label_smoothing: float = 0.05,
        nb_K:            int   = 16,
        nb_heads:        int   = 2,
        nb_attn_dim:     int   = 64,
        layer_attn_proj: int   = 64,
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
        n_nodes     = string_embs.shape[0]
        K           = hp.nb_K
        edge_index_cpu  = graph["edge_index"].cpu()
        edge_weight_cpu = graph["edge_weight"].cpu() if graph["edge_weight"] is not None else torch.ones(edge_index_cpu.shape[1])

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
                nbr_idx_mat[i] = i
                nbr_wgt_mat[i] = 0.0
            elif len(nbrs) <= K:
                n = len(nbrs)
                nbr_idx_mat[i, :n] = torch.tensor(nbrs, dtype=torch.long)
                nbr_wgt_mat[i, :n] = torch.tensor(wgts, dtype=torch.float32)
                if n < K:
                    nbr_idx_mat[i, n:] = i
                    nbr_wgt_mat[i, n:] = 0.0
            else:
                wgt_t = torch.tensor(wgts, dtype=torch.float32)
                nbr_t = torch.tensor(nbrs, dtype=torch.long)
                topk_vals, topk_idx = wgt_t.topk(K)
                nbr_idx_mat[i] = nbr_t[topk_idx]
                nbr_wgt_mat[i] = topk_vals

        self.register_buffer("nbr_idx", nbr_idx_mat)    # [18870, K]
        self.register_buffer("nbr_wgt", nbr_wgt_mat)    # [18870, K]

        # ---- Cross-Layer Attention Fusion module ----
        # Fuses near-final AIDO.Cell layers [16, 17, 18] → 640-dim output
        self.layer_attn = LayerAttentionFusion(
            hidden_dim = HIDDEN_DIM,
            n_layers   = len(FUSION_LAYERS),
            proj_dim   = hp.layer_attn_proj,
        )
        # Cast to float32 (Q/K projections must be float32 for stable optimization)
        self.layer_attn = self.layer_attn.float()

        # ---- Neighborhood Attention Module ----
        self.nb_attn = NeighborhoodAttentionModule(
            in_dim   = STRING_DIM,
            n_heads  = hp.nb_heads,
            attn_dim = hp.nb_attn_dim,
            K        = hp.nb_K,
        )

        # ---- Fusion head ----
        # AIDO cross-layer attention output: 640 (same as single-layer best!)
        # STRING neighborhood: 256
        # Total: 896 — identical to proven best node2-1-1-1-1-1 (F1=0.5128)
        in_dim = HIDDEN_DIM + STRING_DIM  # 640 + 256 = 896
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
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
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

        # ---- AIDO.Cell near-final layer feature extraction ----
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # Need intermediate hidden states
        )
        # hidden_states: tuple of 19 tensors [B, 19266, 640]
        # (index 0 = embedding layer, 1-18 = transformer layers 1-18)
        hidden_states = out.hidden_states

        # Extract summary token (position 19264) from near-final layers [16, 17, 18]
        # All near-final layers have full perturbation context → no shallow noise
        layer_tokens = []
        for layer_idx in FUSION_LAYERS:
            layer_h = hidden_states[layer_idx]              # [B, 19266, 640]
            summary = layer_h[:, AIDO_GENES, :].float()    # [B, 640]  cast to fp32
            layer_tokens.append(summary)

        # Stack: [B, n_layers=3, 640]
        tokens_stacked = torch.stack(layer_tokens, dim=1)  # [B, 3, 640]

        # Cross-layer attention fusion: query=layer18, keys/values=all 3 layers
        # Output: [B, 640] — same dim as single-layer summary token!
        aido_emb = self.layer_attn(tokens_stacked)          # [B, 640]

        # ---- STRING_GNN neighborhood attention ----
        node_indices = torch.tensor(
            [self._string_node_map.get(pid, 0) for pid in pert_ids],
            dtype=torch.long, device=self.string_cache.device
        )  # [B]

        center_emb    = self.string_cache[node_indices]            # [B, 256]
        nbr_node_idx  = self.nbr_idx[node_indices]                 # [B, K]
        nbr_weights   = self.nbr_wgt[node_indices]                 # [B, K]
        neighbor_embs = self.string_cache[nbr_node_idx]            # [B, K, 256]

        string_emb = self.nb_attn(center_emb, neighbor_embs, nbr_weights).float()  # [B, 256]

        # ---- Fusion ----
        fused = torch.cat([aido_emb, string_emb], dim=-1)          # [B, 896]

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
            self.print(f"[Node2-3-1] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()

    # ---- checkpoint: save only trainable params + buffers ----
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
        bufs    = sum(b.numel() for _, b in self.named_buffers())
        self.print(f"Checkpoint: {trained}/{total} params ({100*trained/total:.2f}%), {bufs} buffer values")
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

    parser = argparse.ArgumentParser(
        description="Node2-3-1: AIDO.Cell Cross-Layer Attention Fusion [16,17,18] + STRING K=16 2-head"
    )
    parser.add_argument("--micro-batch-size",   type=int,   default=4)
    parser.add_argument("--global-batch-size",  type=int,   default=32)
    parser.add_argument("--max-epochs",         type=int,   default=250)
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--weight-decay",       type=float, default=2e-2)
    parser.add_argument("--lora-r",             type=int,   default=8)
    parser.add_argument("--lora-alpha",         type=int,   default=16)
    parser.add_argument("--lora-dropout",       type=float, default=0.05)
    parser.add_argument("--head-hidden",        type=int,   default=256)
    parser.add_argument("--head-dropout",       type=float, default=0.5)
    parser.add_argument("--warmup-epochs",      type=int,   default=10)
    parser.add_argument("--label-smoothing",    type=float, default=0.05)
    parser.add_argument("--nb-K",               type=int,   default=16)
    parser.add_argument("--nb-heads",           type=int,   default=2)
    parser.add_argument("--nb-attn-dim",        type=int,   default=64)
    parser.add_argument("--layer-attn-proj",    type=int,   default=64)
    parser.add_argument("--num-workers",        type=int,   default=4)
    parser.add_argument("--val-check-interval", type=float, default=1.0)
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
        lim_test  = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = lim_val = lim_test = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDOLayerAttentionStringFusionModel(
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
        layer_attn_proj = args.layer_attn_proj,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
        auto_insert_metric_name = False,
    )
    es_cb  = EarlyStopping(monitor="val/f1", mode="max", patience=15, min_delta=1e-3)
    lr_cb  = LearningRateMonitor(logging_interval="epoch")
    pg_cb  = TQDMProgressBar(refresh_rate=10)

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

    score_path = Path(__file__).parent / "run" / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node2-3-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
