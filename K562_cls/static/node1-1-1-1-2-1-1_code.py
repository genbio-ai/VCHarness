"""Node 1-1-1-1-2-1-1 – AIDO.Cell-100M LoRA + STRING_GNN K=16 2-Head Neighborhood Attention.

Key architectural innovation: Break the STRING-only ceiling (~0.49 F1) confirmed by parent
(node1-1-1-1-2-1, test F1=0.4913) by adding AIDO.Cell-100M LoRA as a second backbone.

This architecture is based on the proven best-in-tree design (node2-1-1-1-1-1, F1=0.5128):
- AIDO.Cell-100M LoRA (r=8, α=16): fine-tuned on perturbation task via summary token
- STRING_GNN K=16 2-head Neighborhood Attention: frozen pre-computed embeddings with
  attention-weighted context aggregation (2 heads, same as proven best node)
- Simple concatenation fusion: [AIDO summary_token (640) + STRING context (256)] → 896-dim
- 2-layer MLP head: 896 → 256 → 19920 with dropout=0.5
- Weighted cross-entropy + label smoothing ε=0.05

Improvements over the best node (node2-1-1-1-1-1):
1. Extended patience: 20 (vs best node's 10) — feedback: "patience=10 was barely sufficient
   to capture the late spike at epoch 77; increase to 15-20 to reliably capture late-
   improvement phases"
2. max_epochs=200 (unchanged from best node): sufficient budget with patience=20

Parent node (node1-1-1-1-2-1, F1=0.4913) feedback:
- "The confirmed STRING_GNN-only ceiling is ~0.49. The best node (0.5128) uses AIDO.Cell-100M
  LoRA + STRING_GNN."
- "Replace the frozen STRING_GNN backbone with a late fusion of STRING_GNN + scFoundation/
  AIDO.Cell embeddings while retaining this node's proven K=16 neighborhood attention."
- No GenePriorBias: shown to catastrophically hurt AIDO.Cell lineages in multiple nodes
  (node1-1-1-3-1-1: 0.3669, node4-2-2-2: 0.4936 with suboptimal dynamics).

Memory connections:
- node2-1-1-1-1-1 (best in tree, F1=0.5128): same core architecture, patience=10
  → this node extends patience to 20 to improve coverage of late-improvement
- node2-2 (F1=0.5078): "extending patience=20 promising before larger architectural revision"
- node2-1-1-1-1-1 feedback: "increase patience to 15-20 to reliably capture late-improvement"
- node1-1-1-1-2-1 (parent, F1=0.4913): STRING-only ceiling confirmed, must add AIDO.Cell
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import sys
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
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES    = 6640
N_CLASSES  = 3
AIDO_GENES = 19264                        # AIDO.Cell gene vocab size
AIDO_HIDDEN = 640                         # AIDO.Cell-100M hidden dim
STRING_HIDDEN = 256                       # STRING_GNN hidden dim
FUSION_DIM = AIDO_HIDDEN + STRING_HIDDEN  # 896

# Remapped class frequencies (after -1→0, 0→1, 1→2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

MODEL_DIR       = Path("/home/Models/AIDO.Cell-100M")
STRING_GNN_DIR  = Path("/home/Models/STRING_GNN")
DATA_ROOT       = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV       = DATA_ROOT / "train.tsv"
VAL_TSV         = DATA_ROOT / "val.tsv"
TEST_TSV        = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights for weighted cross-entropy."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def load_string_gnn_mapping() -> Dict[str, int]:
    """Load STRING_GNN node_names.json → Ensembl-ID to node-index mapping."""
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    return {name: idx for idx, name in enumerate(node_names)}


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0, 1, 2}
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
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    """K562 DEG prediction dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        string_map: Dict[str, int],
    ) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()

        # STRING_GNN node index for each sample (-1 means not in STRING)
        self.string_node_indices = torch.tensor(
            [string_map.get(p, -1) for p in self.pert_ids], dtype=torch.long
        )

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
            "sample_idx":        idx,
            "pert_id":           self.pert_ids[idx],
            "symbol":            self.symbols[idx],
            "string_node_idx":   self.string_node_indices[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]  # [G] in {0, 1, 2}
        return item


def make_collate_fn(tokenizer):
    """Factory for collate_fn with AIDO.Cell tokenizer."""
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"]  for b in batch]
        symbols  = [b["symbol"]   for b in batch]

        # Tokenize: each sample gets only its perturbed gene with expression=1.0
        # This uses Ensembl IDs via gene_ids; the tokenizer fills other genes with -1.0
        expr_dicts = [
            {"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids
        ]
        tokenized = tokenizer(expr_dicts, return_tensors="pt")  # input_ids: [B, 19264] float32

        out: Dict[str, Any] = {
            "sample_idx":      torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":         pert_ids,
            "symbol":          symbols,
            "input_ids":       tokenized["input_ids"],       # [B, 19264] float32
            "attention_mask":  tokenized["attention_mask"],  # [B, 19264] int64
            "string_node_idx": torch.stack([b["string_node_idx"] for b in batch]),
        }

        # Record which genes are "in vocab" (have expression > -1.0 somewhere)
        # gene_positions: position of perturbed gene in AIDO's 19264-gene vocab
        # (for fallback when gene not found)
        gene_in_vocab = (tokenized["input_ids"] > -1.0).any(dim=1)  # [B]
        gene_positions = (tokenized["input_ids"] > -1.0).float().argmax(dim=1)  # [B]
        out["gene_in_vocab"]  = gene_in_vocab
        out["gene_positions"] = gene_positions

        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out
    return collate_fn


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.string_map: Optional[Dict[str, int]] = None
        self.tokenizer = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.string_map is None:
            self.string_map = load_string_gnn_mapping()

        if self.tokenizer is None:
            # Rank 0 downloads/loads first, then all ranks load
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if local_rank == 0:
                AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df, self.string_map)
        self.val_ds   = DEGDataset(val_df,   self.string_map)
        self.test_ds  = DEGDataset(test_df,  self.string_map)

    def _collate(self):
        return make_collate_fn(self.tokenizer)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=self._collate(), pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self._collate(), pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        sampler = SequentialSampler(self.test_ds)
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self._collate(), pin_memory=True,
            sampler=sampler,
        )


# ---------------------------------------------------------------------------
# Multi-Head Neighborhood Attention Module (STRING_GNN)
# ---------------------------------------------------------------------------
class MultiHeadNeighborhoodAttentionModule(nn.Module):
    """K-hop neighborhood attention with multiple heads for PPI graph context.

    For each perturbed gene, aggregates top-K PPI neighbor embeddings using
    independent attention heads, then projects the concatenated head outputs
    back to the STRING hidden dimension.

    Architecture (per head h):
        q_h = W_q_h(center_emb)                                [B, attn_dim]
        k_h = W_k_h(neigh_embs.reshape(-1, D)).reshape(B, K, attn_dim)
        attn_h = softmax(q_h @ k_h.T / sqrt(attn_dim)) * neigh_weights  [B, K]
        context_h = attn_h @ neigh_embs                        [B, D]
    multi_context = concat([context_h for each head])          [B, n_heads*D]
    output = W_out(multi_context)                              [B, D]

    Proven in node2-1-1-1-1-1 (best in tree, F1=0.5128) with n_heads=2, K=16.
    """

    def __init__(
        self,
        emb_dim: int = STRING_HIDDEN,
        attn_dim: int = 64,
        n_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_heads  = n_heads
        self.attn_dim = attn_dim
        self.W_q = nn.ModuleList([nn.Linear(emb_dim, attn_dim, bias=False) for _ in range(n_heads)])
        self.W_k = nn.ModuleList([nn.Linear(emb_dim, attn_dim, bias=False) for _ in range(n_heads)])
        self.W_out   = nn.Linear(emb_dim * n_heads, emb_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        center_emb: torch.Tensor,   # [B, D]
        neigh_embs: torch.Tensor,   # [B, K, D]
        neigh_weights: torch.Tensor, # [B, K]
        valid_mask: torch.Tensor,   # [B] bool
    ) -> torch.Tensor:
        """Return context-enriched embeddings [B, D]."""
        B, D = center_emb.shape
        K = neigh_embs.shape[1]

        head_contexts = []
        for h in range(self.n_heads):
            q = self.W_q[h](center_emb)   # [B, attn_dim]
            k_flat = self.W_k[h](neigh_embs.reshape(-1, D))
            k = k_flat.reshape(B, K, self.attn_dim)   # [B, K, attn_dim]

            # Scaled dot-product attention: [B, 1, K]
            attn = (q.unsqueeze(1) @ k.transpose(1, 2)) / (self.attn_dim ** 0.5)
            attn = attn.squeeze(1)   # [B, K]

            # Modulate by STRING confidence weights and re-normalize
            attn = F.softmax(attn, dim=-1) * neigh_weights
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            attn = self.dropout(attn)

            ctx = (attn.unsqueeze(1) @ neigh_embs).squeeze(1)  # [B, D]
            head_contexts.append(ctx)

        # Concatenate [B, n_heads*D] → project → [B, D]
        multi_ctx = torch.cat(head_contexts, dim=-1)
        out = self.W_out(multi_ctx)   # [B, D]

        # For unknown pert_ids (no STRING neighbors), return center_emb unchanged
        out = torch.where(valid_mask.unsqueeze(-1), out, center_emb)
        return out


# ---------------------------------------------------------------------------
# Main Model: AIDO.Cell-100M LoRA + STRING_GNN K=16 Multi-Head Fusion
# ---------------------------------------------------------------------------
class AIDOCellStringFusionModel(pl.LightningModule):
    """AIDO.Cell-100M (LoRA r=8) + STRING_GNN K=16 2-head neighborhood attention fusion.

    Architecture:
        1. AIDO.Cell-100M with LoRA (r=8, α=16) → summary token at position AIDO_GENES
           The perturbed gene's expression is set to 1.0; all others are -1.0 (missing)
           Summary token at lhs[:, 19264, :] encodes global perturbation context [B, 640]

        2. STRING_GNN (frozen, pre-computed): lookup + K=16 2-head neighborhood attention
           → enriched perturbation context [B, 256]
           For unknown pert_ids: learnable fallback [B, 256]

        3. Concat fusion: [AIDO summary (640) + STRING context (256)] → [B, 896]

        4. 2-layer MLP head: 896 → 256 → 19920 with dropout=0.5
           logits [B, 3, 6640] via bilinear: einsum("bd,cgd->bcg", h, gene_class_emb)

        5. Weighted cross-entropy + label smoothing ε=0.05
    """

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        head_hidden: int = 256,
        head_dropout: float = 0.5,
        lr_backbone: float = 1e-4,
        lr_head: float = 1e-4,
        weight_decay: float = 2e-2,
        warmup_epochs: int = 10,
        T_max: int = 150,
        label_smoothing: float = 0.05,
        k_neighbors: int = 16,
        attn_dim: int = 64,
        n_attn_heads: int = 2,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def _precompute_topk_neighbors(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        n_nodes: int,
        k: int,
    ):
        """Pre-compute top-K neighbors per node sorted by edge weight."""
        edge_src = edge_index[0].numpy()
        edge_dst = edge_index[1].numpy()
        edge_wt  = edge_weight.numpy()

        order = np.argsort(edge_src, kind="stable")
        edge_src_s = edge_src[order]
        edge_dst_s = edge_dst[order]
        edge_wt_s  = edge_wt[order]

        counts  = np.bincount(edge_src_s, minlength=n_nodes)
        offsets = np.concatenate([[0], np.cumsum(counts)])

        topk_idx_np = np.zeros((n_nodes, k), dtype=np.int64)
        topk_wts_np = np.zeros((n_nodes, k), dtype=np.float32)
        for i in range(n_nodes):
            topk_idx_np[i] = i  # default: self-loop

        for i in range(n_nodes):
            start, end = int(offsets[i]), int(offsets[i + 1])
            if start == end:
                continue
            nbr_dst = edge_dst_s[start:end]
            nbr_wt  = edge_wt_s[start:end]
            n_nbr   = len(nbr_dst)
            k_actual = min(k, n_nbr)
            if n_nbr <= k:
                idx = np.argsort(-nbr_wt)[:k_actual]
            else:
                part_idx = np.argpartition(-nbr_wt, k_actual)[:k_actual]
                idx = part_idx[np.argsort(-nbr_wt[part_idx])]
            topk_idx_np[i, :k_actual] = nbr_dst[idx]
            topk_wts_np[i, :k_actual] = nbr_wt[idx]

        return (
            torch.from_numpy(topk_idx_np).long(),
            torch.from_numpy(topk_wts_np).float(),
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True

        hp = self.hparams

        # ----------------------------------------------------------------
        # 1. AIDO.Cell-100M with LoRA(r=8)
        # ----------------------------------------------------------------
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoModel.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Load model in bf16 to ensure all intermediate activations (including
        # GeneEmbedding output) are bf16 throughout training and evaluation.
        # Without this, Lightning's bf16-mixed autocast does not upcast embedding/lookups,
        # causing hidden_states to be fp32 → LayerNorm outputs fp32 → FA check fails → OOM.
        backbone = AutoModel.from_pretrained(
            str(MODEL_DIR),
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        # NOTE: We deliberately do NOT patch gene_embedding to force requires_grad=True.
        # Gradient checkpointing with use_reentrant=False handles gradient recomputation
        # correctly even when the model input has requires_grad=False — the checkpoint
        # wrapper re-enables grad tracking at the recomputation boundary. Forcing
        # requires_grad=True on gene_embedding's output would keep ALL intermediate
        # activations alive in the autograd graph during the forward pass (PyTorch
        # cannot free them until backward), effectively doubling forward-pass memory.
        # The gene_embedding itself is frozen (requires_grad=False) and its weights
        # will NOT receive gradients anyway, so there is no correctness reason for
        # this patch — only a memory cost.

        # Patch CellFoundationAttention.forward to force bf16 ln_outputs → FlashAttention.
        # Even with model loaded in bf16, during Lightning's sanity check (before training)
        # the autocast context may not be active, causing LN to output fp32 → FA fails → OOM.
        # This patch converts LN output to bf16 as a safety net.
        MODULE_PATH = "transformers_modules.AIDO_dot_Cell_hyphen_100M.modeling_cellfoundation"
        if MODULE_PATH in sys.modules:
            attn_cls = getattr(sys.modules[MODULE_PATH], "CellFoundationAttention", None)
            if attn_cls is not None:
                _orig_attn_forward = attn_cls.forward
                def make_patched_attn_fwd(orig_fwd):
                    def patched_attn_fwd(self, hidden_states, attention_mask=None,
                                         head_mask=None, encoder_hidden_states=None,
                                         encoder_attention_mask=None, past_key_value=None,
                                         output_attentions=False, rotary_pos_emb=None,
                                         use_FA=False):
                        ln_outputs = self.ln(hidden_states)
                        if ln_outputs.dtype == torch.float32 and ln_outputs.is_cuda:
                            ln_outputs = ln_outputs.to(torch.bfloat16)
                        return orig_fwd(self, ln_outputs, attention_mask, head_mask,
                                        encoder_hidden_states, encoder_attention_mask,
                                        past_key_value, output_attentions, rotary_pos_emb, use_FA)
                    return patched_attn_fwd
                attn_cls.forward = make_patched_attn_fwd(_orig_attn_forward)

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            target_modules=["query", "key", "value"],
        )
        self.aido_model = get_peft_model(backbone, lora_cfg)

        # Monkey-patch: AIDO.Cell GeneEmbedding doesn't implement get_input_embeddings(),
        # but PEFT's gradient_checkpointing_enable() calls enable_input_require_grads()
        # which internally calls model.get_input_embeddings().register_forward_hook().
        # We provide a compatible wrapper so the hook can be registered.
        def _compat_get_input_embeddings(model):
            """Return the gene embedding module for enable_input_require_grads compatibility."""
            return model.bert.gene_embedding

        self.aido_model.get_input_embeddings = lambda: _compat_get_input_embeddings(self.aido_model)

        self.aido_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # ----------------------------------------------------------------
        # 2. Pre-compute STRING_GNN embeddings (backbone stays frozen)
        # ----------------------------------------------------------------
        string_backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        string_backbone.eval()
        for p in string_backbone.parameters():
            p.requires_grad = False

        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index  = graph["edge_index"].long()
        edge_weight = graph["edge_weight"].float()

        with torch.no_grad():
            gnn_out  = string_backbone(edge_index=edge_index, edge_weight=edge_weight)
            node_emb = gnn_out.last_hidden_state.float().detach()  # [18870, 256]

        self.register_buffer("node_embeddings", node_emb)  # non-trainable buffer

        # ----------------------------------------------------------------
        # 3. Pre-compute top-K neighbor indices and weights
        # ----------------------------------------------------------------
        n_nodes = node_emb.shape[0]
        topk_idx, topk_wts = self._precompute_topk_neighbors(
            edge_index, edge_weight, n_nodes, hp.k_neighbors
        )
        self.register_buffer("topk_idx", topk_idx)
        self.register_buffer("topk_wts", topk_wts)

        del string_backbone, graph, edge_index, edge_weight, gnn_out

        # ----------------------------------------------------------------
        # 4. Learnable fallback for unknown pert_ids
        # ----------------------------------------------------------------
        self.fallback_string_emb = nn.Embedding(1, STRING_HIDDEN)
        nn.init.normal_(self.fallback_string_emb.weight, std=0.02)

        # ----------------------------------------------------------------
        # 5. Multi-head neighborhood attention (K=16, 2 heads)
        # ----------------------------------------------------------------
        self.neighborhood_attn = MultiHeadNeighborhoodAttentionModule(
            emb_dim=STRING_HIDDEN,
            attn_dim=hp.attn_dim,
            n_heads=hp.n_attn_heads,
            dropout=hp.attn_dropout,
        )

        # ----------------------------------------------------------------
        # 6. 2-layer MLP head + bilinear gene-class embedding
        # ----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Linear(FUSION_DIM, hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
        )
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.head_hidden) * 0.02
        )

        # ----------------------------------------------------------------
        # 7. Class weights for weighted CE
        # ----------------------------------------------------------------
        self.register_buffer("class_weights", get_class_weights())

        # Cast all trainable parameters to float32 for stable optimization
        for _, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Accumulators for val/test
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    def _get_string_embeddings(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Lookup STRING_GNN embeddings + apply K=16 2-head neighborhood attention.

        Args:
            string_node_idx: [B] long tensor, -1 for pert_ids not in STRING.
        Returns:
            [B, STRING_HIDDEN] float32 context embeddings.
        """
        B = string_node_idx.shape[0]
        device = self.node_embeddings.device

        known   = string_node_idx >= 0
        unknown = ~known

        emb = torch.zeros(B, STRING_HIDDEN, dtype=torch.float32, device=device)

        if known.any():
            known_idx  = string_node_idx[known]
            center_emb = self.node_embeddings[known_idx].float()

            neigh_idx = self.topk_idx[known_idx]
            neigh_idx = neigh_idx.clamp(0, self.node_embeddings.shape[0] - 1)
            neigh_embs = self.node_embeddings[neigh_idx.reshape(-1)].float()
            neigh_embs = neigh_embs.reshape(
                known_idx.shape[0], self.hparams.k_neighbors, STRING_HIDDEN
            )
            neigh_wts  = self.topk_wts[known_idx].float()
            valid_mask = torch.ones(known_idx.shape[0], dtype=torch.bool, device=device)

            enriched = self.neighborhood_attn(
                center_emb=center_emb,
                neigh_embs=neigh_embs,
                neigh_weights=neigh_wts,
                valid_mask=valid_mask,
            )
            emb[known] = enriched

        if unknown.any():
            fb = self.fallback_string_emb(
                torch.zeros(unknown.sum(), dtype=torch.long, device=device)
            ).to(torch.float32)
            emb[unknown] = fb

        return emb

    def forward(
        self,
        input_ids: torch.Tensor,         # [B, 19264] float32
        attention_mask: torch.Tensor,    # [B, 19264] int64
        string_node_idx: torch.Tensor,   # [B] long
    ) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        # 1. AIDO.Cell-100M LoRA forward
        aido_out  = self.aido_model(
            input_ids=input_ids.float(),
            attention_mask=attention_mask,
        )
        lhs = aido_out.last_hidden_state  # [B, G+2, 640]

        # Summary token at position AIDO_GENES = 19264 encodes global perturbation context
        summary_emb = lhs[:, AIDO_GENES, :].float()  # [B, 640]

        # 2. STRING_GNN K=16 2-head neighborhood attention
        string_emb = self._get_string_embeddings(string_node_idx)  # [B, 256]

        # 3. Concatenate: [summary_token | STRING_context] → [B, 896]
        fused = torch.cat([summary_emb, string_emb], dim=-1)  # [B, 896]

        # 4. MLP head → [B, head_hidden]
        h = self.head(fused)

        # 5. Bilinear interaction: h · gene_class_emb → [B, 3, G]
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)

        return logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy + mild label smoothing."""
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),
            targets.reshape(-1),
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["string_node_idx"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["string_node_idx"])
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
        local_preds = torch.cat(self._val_preds, dim=0)
        local_tgts  = torch.cat(self._val_tgts,  dim=0)
        local_idx   = torch.cat(self._val_idx,   dim=0)
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds)
        all_tgts  = self.all_gather(local_tgts)
        all_idx   = self.all_gather(local_idx)

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

        order  = torch.argsort(idx_flat)
        s_idx  = idx_flat[order]
        s_pred = preds_flat[order]
        s_tgt  = tgts_flat[order]
        mask   = torch.cat([
            torch.tensor([True], device=s_idx.device),
            s_idx[1:] != s_idx[:-1],
        ])
        preds_dedup = s_pred[mask]
        tgts_dedup  = s_tgt[mask]

        f1 = compute_per_gene_f1(preds_dedup, tgts_dedup)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["string_node_idx"])
        probs  = torch.softmax(logits, dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        """Save test predictions using a standalone sequential dataloader.

        Lightning's trainer.test() wraps even explicit dataloaders with DistributedSampler
        in DDP mode, restricting each rank to num_samples/world_size samples. With 154
        test samples, 2 GPUs, and batch_size=8, this gives ~77 samples per rank → only
        128 total after deduplication (last 26 samples missing).

        This method bypasses Lightning's test dataloader entirely by iterating the test
        dataset directly with a SequentialSampler, guaranteeing all 154 samples are processed.
        The model's weights (from fit() or loaded from best checkpoint) are used directly.
        """
        # Use a fresh SequentialSampler to guarantee ALL 154 samples are processed
        # regardless of what Lightning's test dataloader does
        from torch.utils.data import DataLoader, SequentialSampler
        self.print("[on_test_epoch_end] Starting sequential test loop")
        self.eval()

        # Get datamodule from trainer
        dm = self.trainer.datamodule
        test_ds = dm.test_ds
        collate_fn = dm._collate()
        test_dl = DataLoader(
            test_ds,
            batch_size=8,
            shuffle=False,
            sampler=SequentialSampler(test_ds),
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        self.print(f"[on_test_epoch_end] Test dataloader: {len(test_dl)} batches, {len(test_ds)} samples")

        all_preds: list[torch.Tensor] = []
        all_idx: list[torch.Tensor] = []

        device = next(self.parameters()).device
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for batch in test_dl:
                batch_gpu = {
                    "input_ids": batch["input_ids"].to(device, non_blocking=True),
                    "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
                    "string_node_idx": batch["string_node_idx"].to(device, non_blocking=True),
                }
                logits = self(
                    batch_gpu["input_ids"],
                    batch_gpu["attention_mask"],
                    batch_gpu["string_node_idx"],
                )
                probs = torch.softmax(logits, dim=1).float()
                all_preds.append(probs.cpu())
                all_idx.append(batch["sample_idx"])

        self.print(f"[on_test_epoch_end] Collected {len(all_preds)} batches")

        # Gather from all ranks (both process same samples via SequentialSampler)
        local_preds_t = torch.cat(all_preds, dim=0)
        local_idx_t = torch.cat(all_idx, dim=0)

        gathered_preds = self.all_gather(local_preds_t)
        gathered_idx = self.all_gather(local_idx_t)

        if self.trainer.is_global_zero:
            preds_flat = gathered_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat = gathered_idx.view(-1)

            # Deduplicate: keep first occurrence of each sample index
            pred_map: Dict[int, torch.Tensor] = {}
            for i in range(len(idx_flat)):
                gid = int(idx_flat[i].item())
                if gid not in pred_map:
                    pred_map[gid] = preds_flat[i]

            self.print(f"[on_test_epoch_end] Unique predictions: {len(pred_map)}")

            test_df = pd.read_csv(TEST_TSV, sep="\t")
            rows = []
            for i in range(len(test_df)):
                if i not in pred_map:
                    self.print(f"[on_test_epoch_end] WARNING: missing prediction for test index {i}")
                    continue
                pid = test_df.iloc[i]["pert_id"]
                sym = test_df.iloc[i]["symbol"]
                pred = pred_map[i].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            self.print(f"[on_test_epoch_end] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()

    # ------------------------------------------------------------------
    # Checkpoint helpers — save only trainable params + buffers
    # ------------------------------------------------------------------
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
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Checkpoint: {train}/{total} params ({100 * train / total:.1f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Optimizer: AdamW + linear warmup + CosineAnnealingLR
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=hp.lr_backbone, weight_decay=hp.weight_decay)

        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=hp.warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=hp.T_max, eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[hp.warmup_epochs],
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node1-1-1-1-2-1-1 – AIDO.Cell-100M LoRA + STRING_GNN K=16 2-Head Fusion"
    )
    parser.add_argument("--micro-batch-size",  type=int,   default=4)
    parser.add_argument("--global-batch-size", type=int,   default=256)
    parser.add_argument("--max-epochs",        type=int,   default=200)
    parser.add_argument("--lr-backbone",       type=float, default=1e-4,  dest="lr_backbone")
    parser.add_argument("--lr-head",           type=float, default=1e-4,  dest="lr_head")
    parser.add_argument("--weight-decay",      type=float, default=2e-2)
    parser.add_argument("--head-hidden",       type=int,   default=256,   dest="head_hidden")
    parser.add_argument("--head-dropout",      type=float, default=0.5,   dest="head_dropout")
    parser.add_argument("--warmup-epochs",     type=int,   default=10)
    parser.add_argument("--t-max",             type=int,   default=150,   dest="t_max")
    parser.add_argument("--label-smoothing",   type=float, default=0.05,  dest="label_smoothing")
    parser.add_argument("--lora-r",            type=int,   default=8,     dest="lora_r")
    parser.add_argument("--lora-alpha",        type=int,   default=16,    dest="lora_alpha")
    parser.add_argument("--lora-dropout",      type=float, default=0.05,  dest="lora_dropout")
    parser.add_argument("--k-neighbors",       type=int,   default=16,    dest="k_neighbors")
    parser.add_argument("--attn-dim",          type=int,   default=64,    dest="attn_dim")
    parser.add_argument("--n-attn-heads",      type=int,   default=2,     dest="n_attn_heads")
    parser.add_argument("--attn-dropout",      type=float, default=0.1,   dest="attn_dropout")
    parser.add_argument("--patience",          type=int,   default=20)
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--debug-max-step",    type=int,   default=None,  dest="debug_max_step")
    parser.add_argument("--fast-dev-run",      action="store_true",       dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    # For fast_dev_run, reduce micro_batch_size to 2 to avoid OOM
    # (Lightning's fast_dev_run processes 1 batch with the configured batch_size).
    if fast_dev_run:
        args.micro_batch_size = 2

    if args.debug_max_step is not None:
        lim_train = args.debug_max_step
        lim_val   = args.debug_max_step
        lim_test  = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = 1.0
        lim_val   = 1.0
        lim_test  = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # DataModule
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    # Model
    model = AIDOCellStringFusionModel(
        lora_r           = args.lora_r,
        lora_alpha       = args.lora_alpha,
        lora_dropout     = args.lora_dropout,
        head_hidden      = args.head_hidden,
        head_dropout     = args.head_dropout,
        lr_backbone      = args.lr_backbone,
        lr_head          = args.lr_head,
        weight_decay     = args.weight_decay,
        warmup_epochs    = args.warmup_epochs,
        T_max            = args.t_max,
        label_smoothing  = args.label_smoothing,
        k_neighbors      = args.k_neighbors,
        attn_dim         = args.attn_dim,
        n_attn_heads     = args.n_attn_heads,
        attn_dropout     = args.attn_dropout,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
    )
    es_cb = EarlyStopping(
        monitor   = "val/f1",
        mode      = "max",
        patience  = args.patience,   # 20 (extended from best node's 10)
        min_delta = 1e-4,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy: DDP for multi-GPU, auto for single-GPU / fast_dev_run
    use_ddp = n_gpus > 1 and not fast_dev_run
    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if use_ddp else "auto"
    )
    devices_for_trainer = 1 if (fast_dev_run and n_gpus > 1) else n_gpus

    trainer = pl.Trainer(
        accelerator             = "gpu",
        devices                 = devices_for_trainer,
        num_nodes               = 1,
        strategy                = strategy,
        precision               = "bf16-mixed",
        max_epochs              = args.max_epochs,
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

    # Run test using trainer.test() which triggers on_test_epoch_end where we
    # run the sequential test loop (bypasses Lightning's DistributedSampler).
    # For debug runs, use current model weights; for full runs, use best checkpoint.
    ckpt_path = "best" if (args.debug_max_step is None and not fast_dev_run) else None
    if ckpt_path:
        test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)
    else:
        test_results = trainer.test(model, datamodule=dm)

    # Save test score
    score_path = output_dir / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node1-1-1-1-2-1-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
