"""Node 3-1-2-3: AIDO.Cell-100M LoRA r=8 + STRING_GNN K=16 2-head + Bilinear Gene-Class Head.

Primary innovation over siblings (node3-1-2-1 and node3-1-2-2):
- Uses bilinear gene-class embedding head (gene_class_emb [3, 6640, 256]) instead of linear head
- Proven: node2-1-1-1-1-1 (bilinear, F1=0.5128) vs node1-1-1-1-1-2 (linear, F1=0.4748) = +0.038 F1
- Uses 2-head neighborhood attention (vs 1-head in siblings) — matches tree-best node config
- Uses global_batch=256 (vs 32-128 in siblings) — matches node2 lineage proven recipe
- AIDO.Cell-100M LoRA r=8 (same as sibling 2 but with bilinear head)
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
N_GENES    = 6640
N_CLASSES  = 3
AIDO_GENES = 19264
AIDO_MODEL_DIR    = "/home/Models/AIDO.Cell-100M"
STRING_GNN_DIR    = "/home/Models/STRING_GNN"
HIDDEN_DIM = 640      # AIDO.Cell-100M hidden size
STRING_DIM = 256      # STRING_GNN embedding dimension

# Class frequency for class weight computation (down, neutral, up)
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency class weights for class imbalance."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute per-gene macro F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, N_GENES] float32 softmax probabilities
        targets: [N, N_GENES] int64 labels in {0,1,2}
    Returns:
        Scalar mean F1 over all genes.
    """
    y_hat       = preds.argmax(dim=1)  # [N, N_GENES]
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


def load_string_gnn_embeddings_and_graph() -> Tuple[torch.Tensor, Dict[str, int], torch.Tensor, torch.Tensor]:
    """Load frozen STRING_GNN embeddings and pre-compute K-neighbor index/weight tables.

    Returns:
        emb_matrix:  [18870, 256] float32 per-gene PPI embeddings
        name_to_idx: dict mapping Ensembl gene ID -> row index in emb_matrix
        edge_index:  [2, E] long tensor for building neighbor lookup
        edge_weight: [E] float tensor of STRING confidence scores
    """
    node_names = json.loads((Path(STRING_GNN_DIR) / "node_names.json").read_text())
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_model = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True).to(device)
    gnn_model.eval()

    graph = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", map_location=device)
    edge_index  = graph["edge_index"]
    edge_weight = graph.get("edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    with torch.no_grad():
        outputs = gnn_model(edge_index=edge_index, edge_weight=edge_weight)
        emb_matrix = outputs.last_hidden_state.float().cpu()  # [18870, 256]

    # Keep edge_index and edge_weight on CPU for neighbor lookup
    edge_index_cpu  = edge_index.cpu()
    edge_weight_cpu = edge_weight.cpu() if edge_weight is not None else None

    del gnn_model
    torch.cuda.empty_cache()

    return emb_matrix, name_to_idx, edge_index_cpu, edge_weight_cpu


def build_top_k_neighbors(
    emb_matrix: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    K: int = 16,
    n_nodes: int = 18870,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute top-K neighbor indices and weights for all STRING nodes.

    Returns:
        topk_idx:     [n_nodes, K] long tensor of neighbor node indices
        topk_weights: [n_nodes, K] float tensor of edge weights (normalized)
    """
    # Build adjacency from edge_index: for each source node, collect (dest, weight)
    src_nodes = edge_index[0]  # [E]
    dst_nodes = edge_index[1]  # [E]

    if edge_weight is None:
        edge_weight = torch.ones(src_nodes.shape[0])

    # Initialize outputs with self-loops (fallback when neighbor count < K)
    topk_idx     = torch.arange(n_nodes).unsqueeze(1).expand(n_nodes, K).clone()
    topk_weights = torch.zeros(n_nodes, K)

    # Build per-node neighbor dict efficiently
    # Using scatter approach: group by source node
    adj: Dict[int, List[Tuple[int, float]]] = {}
    for e_idx in range(src_nodes.shape[0]):
        s = src_nodes[e_idx].item()
        d = dst_nodes[e_idx].item()
        w = edge_weight[e_idx].item()
        if s not in adj:
            adj[s] = []
        adj[s].append((d, w))

    for node_id, neighbors in adj.items():
        if not neighbors:
            continue
        # Sort by weight descending, take top K
        neighbors_sorted = sorted(neighbors, key=lambda x: -x[1])[:K]
        n_actual = len(neighbors_sorted)
        for k_pos, (nbr_id, nbr_w) in enumerate(neighbors_sorted):
            topk_idx[node_id, k_pos]     = nbr_id
            topk_weights[node_id, k_pos] = nbr_w
        # Fill remaining with self-loop (zero weight) — already initialized

    # Normalize weights to [0, 1] range
    max_w = topk_weights.max()
    if max_w > 0:
        topk_weights = topk_weights / max_w

    return topk_idx, topk_weights


# ---------------------------------------------------------------------------
# Two-Head Neighborhood Attention Module
# ---------------------------------------------------------------------------
class TwoHeadNeighborhoodAttention(nn.Module):
    """Two-head PPI neighborhood attention aggregation for STRING_GNN embeddings.

    Architecture:
        - 2 independent attention heads, each with attn_dim=64
        - Center + neighbor concat -> linear -> scalar attention score + edge_weight bias
        - Softmax over neighbors per head
        - Concatenate both head outputs [B, 2*256=512] -> project to [B, 256]
        - Center-context gate: sigmoid(Linear(center_emb)) * aggregated_context
        - Output: center_emb + gate * projected_context  # residual connection

    Proven in: node2-1-1-1-1-1 (F1=0.5128, tree best)
    """

    def __init__(
        self,
        emb_dim: int = 256,
        n_heads: int = 2,
        attn_dim: int = 64,
        K: int = 16,
    ) -> None:
        super().__init__()
        self.emb_dim  = emb_dim
        self.n_heads  = n_heads
        self.attn_dim = attn_dim
        self.K        = K

        # Per-head attention score projections: (center + neighbor) -> scalar
        self.attn_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * emb_dim, attn_dim),
                nn.Tanh(),
                nn.Linear(attn_dim, 1),
            )
            for _ in range(n_heads)
        ])

        # Aggregate projection: concatenate all head outputs [n_heads * emb_dim] -> emb_dim
        self.agg_proj = nn.Linear(n_heads * emb_dim, emb_dim)

        # Gate: center -> [emb_dim] sigmoid gate
        self.gate_proj = nn.Linear(emb_dim, emb_dim)

    def forward(
        self,
        center_emb:   torch.Tensor,  # [B, emb_dim]
        topk_emb:     torch.Tensor,  # [B, K, emb_dim]
        topk_weights: torch.Tensor,  # [B, K] normalized edge weights
    ) -> torch.Tensor:
        """Compute neighborhood-aggregated STRING embedding.

        Returns: [B, emb_dim] context-enriched embedding.
        """
        B, K, D = topk_emb.shape

        # Expand center for concat with each neighbor
        center_exp = center_emb.unsqueeze(1).expand(B, K, D)  # [B, K, D]
        pair = torch.cat([center_exp, topk_emb], dim=-1)       # [B, K, 2*D]

        head_outputs = []
        for head_proj in self.attn_projs:
            # Compute attention scores
            scores = head_proj(pair).squeeze(-1)          # [B, K]
            scores = scores + topk_weights                # add edge weight bias
            weights = torch.softmax(scores, dim=-1)       # [B, K]
            # Weighted sum of neighbor embeddings
            agg = (weights.unsqueeze(-1) * topk_emb).sum(dim=1)  # [B, D]
            head_outputs.append(agg)

        # Concat all heads and project
        multi_head = torch.cat(head_outputs, dim=-1)      # [B, n_heads*D]
        ctx = self.agg_proj(multi_head)                   # [B, D]

        # Gated residual connection
        gate = torch.sigmoid(self.gate_proj(center_emb))  # [B, D]
        output = center_emb + gate * ctx                  # [B, D]

        return output


# ---------------------------------------------------------------------------
# Dataset / DataModule
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        string_emb: torch.Tensor,
        name_to_idx: Dict[str, int],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        self.pert_ids    = df["pert_id"].tolist()
        self.symbols     = df["symbol"].tolist()
        self.string_emb  = string_emb      # [18870, 256]
        self.name_to_idx = name_to_idx
        self.topk_idx    = topk_idx        # [18870, K]
        self.topk_weights = topk_weights   # [18870, K]

        has_label = "label" in df.columns and df["label"].notna().all()
        self.labels: Optional[List] = (
            [torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
             for row in df["label"].tolist()]
            if has_label else None
        )

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid = self.pert_ids[idx]
        gnn_node_idx = self.name_to_idx.get(pid, -1)

        if gnn_node_idx >= 0:
            center_emb    = self.string_emb[gnn_node_idx]           # [256]
            nbr_indices   = self.topk_idx[gnn_node_idx]             # [K]
            nbr_weights   = self.topk_weights[gnn_node_idx]         # [K]
            nbr_emb       = self.string_emb[nbr_indices]            # [K, 256]
        else:
            center_emb  = torch.zeros(STRING_DIM)
            nbr_emb     = torch.zeros(self.topk_idx.shape[1], STRING_DIM)
            nbr_weights = torch.zeros(self.topk_idx.shape[1])

        item: Dict[str, Any] = {
            "sample_idx":  idx,
            "pert_id":     pid,
            "symbol":      self.symbols[idx],
            "center_emb":  center_emb,    # [256]
            "nbr_emb":     nbr_emb,       # [K, 256]
            "nbr_weights": nbr_weights,   # [K]
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def make_collate(tokenizer):
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]

        # Tokenize for AIDO.Cell: single gene with expression=1.0
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        input_ids  = tokenized["input_ids"]   # [B, 19264] float32

        out: Dict[str, Any] = {
            "sample_idx":     torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":        pert_ids,
            "symbol":         symbols,
            "input_ids":      input_ids,
            "attention_mask": tokenized["attention_mask"],
            "center_emb":     torch.stack([b["center_emb"]  for b in batch]),   # [B, 256]
            "nbr_emb":        torch.stack([b["nbr_emb"]     for b in batch]),   # [B, K, 256]
            "nbr_weights":    torch.stack([b["nbr_weights"] for b in batch]),   # [B, K]
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out
    return collate_fn


class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.tokenizer   = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Load tokenizer (rank-0 first to avoid race conditions)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        # Load frozen STRING_GNN embeddings once and pre-compute neighbor tables
        emb_matrix, name_to_idx, edge_index, edge_weight = load_string_gnn_embeddings_and_graph()

        print("[Node3-1-2-3] Pre-computing top-K neighbor tables...")
        K = 16
        topk_idx, topk_weights = build_top_k_neighbors(
            emb_matrix, edge_index, edge_weight, K=K, n_nodes=emb_matrix.shape[0]
        )
        print(f"[Node3-1-2-3] topk_idx shape: {topk_idx.shape}, topk_weights shape: {topk_weights.shape}")

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df, emb_matrix, name_to_idx, topk_idx, topk_weights)
        self.val_ds   = DEGDataset(val_df,   emb_matrix, name_to_idx, topk_idx, topk_weights)
        self.test_ds  = DEGDataset(test_df,  emb_matrix, name_to_idx, topk_idx, topk_weights)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_collate(self.tokenizer),
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BilinearHead(nn.Module):
    """Bilinear factorized classification head.

    Computes per-gene per-class logits via bilinear interaction:
        logits[b, c, g] = dot(proj(h)[b], gene_class_emb[c, g])

    Architecture proven in node2-1-1-1-1-1 (F1=0.5128, tree best).

    Args:
        input_dim:    dimension of the fused representation
        bilinear_dim: latent dim for gene-class embeddings
        n_classes:    3 (down/neutral/up)
        n_genes:      6640
        dropout:      dropout rate before projection
    """

    def __init__(
        self,
        input_dim: int    = 896,   # 640 AIDO + 256 STRING
        bilinear_dim: int = 256,
        n_classes: int    = N_CLASSES,
        n_genes: int      = N_GENES,
        dropout: float    = 0.5,
    ) -> None:
        super().__init__()
        self.dropout    = nn.Dropout(dropout)
        self.proj       = nn.Linear(input_dim, bilinear_dim)
        # Learnable gene-class embedding table: [n_classes, n_genes, bilinear_dim]
        self.gene_class_emb = nn.Parameter(
            torch.randn(n_classes, n_genes, bilinear_dim) * 0.01
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, input_dim] fused representation
        Returns:
            logits: [B, n_classes, n_genes]
        """
        h = self.dropout(h)
        h_proj = self.proj(h)  # [B, bilinear_dim]
        # Bilinear: [B, bilinear_dim] x [n_classes, n_genes, bilinear_dim]
        # -> logits[b, c, g] = sum_d h_proj[b, d] * gene_class_emb[c, g, d]
        logits = torch.einsum("bd,cgd->bcg", h_proj, self.gene_class_emb)
        return logits


class AIDOStringBilinearModel(pl.LightningModule):
    """AIDO.Cell-100M LoRA r=8 + STRING_GNN K=16 2-head + Bilinear gene-class head.

    Architecture:
    - AIDO.Cell-100M with LoRA r=8 (all 18 layers, QKV projections)
    - Summary token extraction (position 19264 of last hidden state) -> [B, 640]
    - Frozen STRING_GNN + K=16 2-head neighborhood attention -> [B, 256]
    - Concatenation fusion: [B, 640+256=896]
    - Bilinear classification head: proj(896→256) x gene_class_emb[3,6640,256] -> [B, 3, 6640]
    - Label-smoothed CE + sqrt-inverse-frequency class weights
    - AdamW, lr=1e-4, weight_decay=2e-2
    - 10-epoch linear warmup + CosineAnnealingLR(T_max=200)

    Key innovation vs siblings:
    - node3-1-2-1: uses AIDO-10M + simple linear head
    - node3-1-2-2: uses AIDO-100M LoRA + simple linear head
    - THIS NODE: AIDO-100M LoRA + 2-head + BILINEAR head (proven +0.038 F1 vs linear)
    """

    def __init__(
        self,
        bilinear_dim: int   = 256,
        head_dropout: float = 0.5,
        lr: float           = 1e-4,
        weight_decay: float = 2e-2,
        warmup_epochs: int  = 10,
        t_max: int          = 200,
        eta_min_ratio: float = 1e-2,
        label_smoothing: float = 0.05,
        n_attn_heads: int    = 2,
        attn_dim: int        = 64,
        K: int               = 16,
        lora_r: int          = 8,
        lora_alpha: int      = 16,
        lora_dropout: float  = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load AIDO.Cell-100M with LoRA ----
        base_model = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        base_model = base_model.to(torch.bfloat16)
        base_model.config.use_cache = False

        # Enable gradient checkpointing BEFORE PEFT wrapping (reduces activation memory)
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # CRITICAL OOM FIX: Monkey-patch bert_extended_attention_mask to avoid [B,S,S]
        # attention mask computation (S=19264, B=32 => 95 GB in int64). FlashAttention
        # ignores this mask (assumes no padding). Returning scalar zero is safe:
        # - FlashAttention: mask is received but never used
        # - Standard attention (backup): adds 0 to scores = no effect
        import sys as _sys
        _cf_mod = _sys.modules.get(
            "transformers_modules.AIDO_dot_Cell_hyphen_100M.modeling_cellfoundation"
        )
        if _cf_mod is not None:
            def _noop_extended_mask(attention_mask):
                return torch.zeros(1, device=attention_mask.device, dtype=torch.float32)
            _cf_mod.bert_extended_attention_mask = _noop_extended_mask

        # Share QKV weights between flash_self and self attention modules so that
        # FlashAttention uses LoRA-updated weights (proven fix from sibling node3-1-2-2)
        for layer in base_model.bert.encoder.layer:
            ss = layer.attention.flash_self
            mm = layer.attention.self
            ss.query.weight = mm.query.weight
            ss.key.weight   = mm.key.weight
            ss.value.weight = mm.value.weight
            if hasattr(mm.query, "bias") and mm.query.bias is not None:
                ss.query.bias = mm.query.bias
                ss.key.bias   = mm.key.bias
                ss.value.bias = mm.value.bias

        # CRITICAL FIX: Monkey-patch enable_input_require_grads (PEFT compatibility).
        # AIDO.Cell does not implement get_input_embeddings(), so PEFT's internal
        # _prepare_model_for_gradient_checkpointing -> enable_input_require_grads()
        # raises NotImplementedError. Monkey-patch to bypass, then register hook manually.
        # (proven fix from node2-1-1-1-1-1 and sibling node3-1-2-2)
        base_model.enable_input_require_grads = lambda: None

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,  # all 18 layers
        )
        self.aido_backbone = get_peft_model(base_model, lora_cfg)
        self.aido_backbone.print_trainable_parameters()

        # Register gradient hook on gene_embedding so LoRA adapters receive gradients
        # (proven fix from node2-1-1-1-1-1 and sibling node3-1-2-2)
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        self.aido_backbone.model.bert.gene_embedding.register_forward_hook(
            _make_inputs_require_grad
        )

        # Cast LoRA params to float32 for stable optimization
        for name, param in self.aido_backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        total_params = sum(p.numel() for p in self.aido_backbone.parameters())
        trainable_params = sum(p.numel() for p in self.aido_backbone.parameters() if p.requires_grad)
        print(f"[Node3-1-2-3] AIDO-100M LoRA trainable: {trainable_params:,} / {total_params:,}")

        # ---- Two-head STRING_GNN neighborhood attention ----
        self.string_attn = TwoHeadNeighborhoodAttention(
            emb_dim  = STRING_DIM,
            n_heads  = hp.n_attn_heads,
            attn_dim = hp.attn_dim,
            K        = hp.K,
        )
        # Cast to float32
        for p in self.string_attn.parameters():
            p.data = p.data.float()

        # ---- Bilinear gene-class head ----
        # Input: AIDO 640-dim (summary token) + STRING 256-dim = 896-dim
        fusion_dim = HIDDEN_DIM + STRING_DIM  # 640 + 256 = 896
        self.head = BilinearHead(
            input_dim    = fusion_dim,
            bilinear_dim = hp.bilinear_dim,
            n_classes    = N_CLASSES,
            n_genes      = N_GENES,
            dropout      = hp.head_dropout,
        )
        # Cast to float32
        for p in self.head.parameters():
            p.data = p.data.float()

        # ---- Loss ----
        class_weights = get_class_weights()
        self.register_buffer("class_weights", class_weights)

        # ---- Accumulators for validation/test ----
        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds: List[torch.Tensor] = []
        self._test_meta:  List[Tuple]        = []

    # ---- Forward ----
    def forward(
        self,
        input_ids:      torch.Tensor,  # [B, 19264] float32
        attention_mask: torch.Tensor,  # [B, 19264]
        center_emb:     torch.Tensor,  # [B, 256]
        nbr_emb:        torch.Tensor,  # [B, K, 256]
        nbr_weights:    torch.Tensor,  # [B, K]
    ) -> torch.Tensor:
        B = input_ids.shape[0]

        # AIDO.Cell-100M forward pass
        out = self.aido_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Summary token at position 19264 (first appended summary position)
        # last_hidden_state shape: [B, 19266, 640]
        aido_feat = out.last_hidden_state[:, 19264, :].float()  # [B, 640]

        # STRING_GNN 2-head neighborhood attention
        center_f    = center_emb.float().to(aido_feat.device)
        nbr_f       = nbr_emb.float().to(aido_feat.device)
        nbr_w       = nbr_weights.float().to(aido_feat.device)
        string_feat = self.string_attn(center_f, nbr_f, nbr_w)  # [B, 256]

        # Fused representation: [B, 896]
        fused = torch.cat([aido_feat, string_feat], dim=-1)

        # Bilinear classification: [B, 3, 6640]
        logits = self.head(fused)
        return logits

    # ---- Loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        flat_logits  = logits.permute(0, 2, 1).reshape(-1, C)
        flat_targets = targets.reshape(-1)
        return F.cross_entropy(
            flat_logits, flat_targets,
            weight=self.class_weights.to(flat_logits.device),
            label_smoothing=self.hparams.label_smoothing,
        )

    # ---- Training ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["center_emb"], batch["nbr_emb"], batch["nbr_weights"],
        )
        loss = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    # ---- Validation ----
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["center_emb"], batch["nbr_emb"], batch["nbr_weights"],
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
        self._val_preds.clear()
        self._val_tgts.clear()
        self._val_idx.clear()

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
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    # ---- Test ----
    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["center_emb"], batch["nbr_emb"], batch["nbr_weights"],
        )
        probs = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        for i, (pid, sym) in enumerate(zip(batch["pert_id"], batch["symbol"])):
            self._test_meta.append((pid, sym, batch["sample_idx"][i].item()))
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds = torch.cat(self._test_preds, 0)  # [local_N, 3, 6640]
        local_rows  = [
            {
                "sample_idx": sidx,
                "pert_id":    pid,
                "symbol":     sym,
                "prediction": local_preds[i].cpu().numpy().tolist(),
            }
            for i, (pid, sym, sidx) in enumerate(self._test_meta)
        ]

        # Gather all rows from all ranks via all_gather_object
        if self.trainer.is_global_zero:
            if torch.distributed.is_initialized():
                world_size   = torch.distributed.get_world_size()
                all_meta_obj = [None] * world_size
                torch.distributed.all_gather_object(all_meta_obj, local_rows)
            else:
                all_meta_obj = [local_rows]
        else:
            if torch.distributed.is_initialized():
                _dummy = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(_dummy, local_rows)

        self._test_preds.clear()
        self._test_meta.clear()

        if not self.trainer.is_global_zero:
            return

        # Flatten from all ranks and deduplicate by sample_idx
        global_rows = []
        for rank_rows in all_meta_obj:
            global_rows.extend(rank_rows)

        seen = set()
        unique_rows = []
        for row in global_rows:
            if row["sample_idx"] not in seen:
                seen.add(row["sample_idx"])
                unique_rows.append(row)

        rows = [
            {
                "idx":        row["pert_id"],
                "input":      row["symbol"],
                "prediction": json.dumps(row["prediction"]),
            }
            for row in unique_rows
        ]

        out_dir = Path(__file__).parent / "run"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
        print(f"[Node3-1-2-3] Saved {len(rows)} test predictions to {out_dir / 'test_predictions.tsv'}")

    # ---- Checkpoint: save only trainable parameters ----
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
        self.print(f"[Node3-1-2-3] Checkpoint: {trained:,}/{total:,} params ({100*trained/total:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer ----
    def configure_optimizers(self):
        hp = self.hparams

        # Single AdamW optimizer for all trainable parameters
        # (LoRA adapters + STRING attention module + bilinear head)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=hp.lr,
            weight_decay=hp.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Linear warmup + CosineAnnealingLR schedule
        # Proven in node2-1-1-1-1-1 (F1=0.5128)
        def warmup_cosine_lr(epoch: int) -> float:
            if epoch < hp.warmup_epochs:
                return float(epoch + 1) / float(hp.warmup_epochs)
            progress = (epoch - hp.warmup_epochs) / max(1, hp.t_max - hp.warmup_epochs)
            return max(
                hp.eta_min_ratio,
                0.5 * (1.0 + np.cos(np.pi * progress)),
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lr)

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
        description="Node3-1-2-3: AIDO.Cell-100M LoRA + STRING K=16 2-head + Bilinear Head"
    )
    parser.add_argument("--micro-batch-size",  type=int,   default=4)
    parser.add_argument("--global-batch-size", type=int,   default=256)
    parser.add_argument("--max-epochs",        type=int,   default=250)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--weight-decay",      type=float, default=2e-2)
    parser.add_argument("--bilinear-dim",      type=int,   default=256)
    parser.add_argument("--head-dropout",      type=float, default=0.5)
    parser.add_argument("--warmup-epochs",     type=int,   default=10)
    parser.add_argument("--t-max",             type=int,   default=200)
    parser.add_argument("--eta-min-ratio",     type=float, default=1e-2)
    parser.add_argument("--label-smoothing",   type=float, default=0.05)
    parser.add_argument("--n-attn-heads",      type=int,   default=2)
    parser.add_argument("--attn-dim",          type=int,   default=64)
    parser.add_argument("--K",                 type=int,   default=16)
    parser.add_argument("--lora-r",            type=int,   default=8)
    parser.add_argument("--lora-alpha",        type=int,   default=16)
    parser.add_argument("--lora-dropout",      type=float, default=0.05)
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--patience",          type=int,   default=15)
    parser.add_argument("--debug_max_step",    type=int,   default=None)
    parser.add_argument("--fast_dev_run",      action="store_true")
    parser.add_argument("--val-check-interval", type=float, default=1.0)
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

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDOStringBilinearModel(
        bilinear_dim     = args.bilinear_dim,
        head_dropout     = args.head_dropout,
        lr               = args.lr,
        weight_decay     = args.weight_decay,
        warmup_epochs    = args.warmup_epochs,
        t_max            = args.t_max,
        eta_min_ratio    = args.eta_min_ratio,
        label_smoothing  = args.label_smoothing,
        n_attn_heads     = args.n_attn_heads,
        attn_dim         = args.attn_dim,
        K                = args.K,
        lora_r           = args.lora_r,
        lora_alpha       = args.lora_alpha,
        lora_dropout     = args.lora_dropout,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1", mode="max", save_top_k=1,
    )
    es_cb  = EarlyStopping(monitor="val/f1", mode="max", patience=args.patience, min_delta=1e-4)
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
        val_check_interval      = args.val_check_interval if (args.debug_max_step is None and not fast_dev_run) else 1.0,
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

    # Compute real F1 from saved test predictions using calc_metric.py
    score_path = Path(__file__).parent / "test_score.txt"
    pred_path  = Path(__file__).parent / "run" / "test_predictions.tsv"
    if pred_path.exists() and TEST_TSV.exists():
        import subprocess
        try:
            result = subprocess.run(
                ["python", str(DATA_ROOT / "calc_metric.py"), str(pred_path), str(TEST_TSV)],
                capture_output=True, text=True, timeout=180,
            )
            output_str = result.stdout.strip()
            if output_str:
                metrics = json.loads(output_str.split("\n")[-1])
                f1_score = metrics.get("value", None)
                if f1_score is not None:
                    with open(score_path, "w") as f:
                        f.write(f"f1_score: {f1_score}\n")
                        if "details" in metrics:
                            for k, v in metrics["details"].items():
                                f.write(f"  {k}: {v}\n")
                    print(f"[Node3-1-2-3] test_f1={f1_score:.4f} — saved to {score_path}")
                else:
                    with open(score_path, "w") as f:
                        f.write(f"error: {metrics.get('error', 'no value key')}\n")
            else:
                with open(score_path, "w") as f:
                    f.write(f"error: empty output from calc_metric\nstderr: {result.stderr[:500]}\n")
        except Exception as e:
            with open(score_path, "w") as f:
                f.write(f"error: {e}\n")
            print(f"[Node3-1-2-3] calc_metric failed: {e}")
    else:
        with open(score_path, "w") as f:
            f.write("error: test_predictions.tsv not found\n")
    print(f"[Node3-1-2-3] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
