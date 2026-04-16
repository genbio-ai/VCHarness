"""Node 2-2-1-1-1-1: AIDO.Cell-100M LoRA(r=8) + Intra-STRING Neighborhood Attention
         + Discriminative LRs + Extended Training (Regression Recovery)

Parent: node2-2-1-1-1 (test F1=0.4891, regression of -0.0219 from node2-2-1-1)

Key improvements over parent (node2-2-1-1-1):

1. Revert CrossModalNeighborhoodAttention -> NeighborhoodAttentionModule (intra-STRING):
   - Parent's cross-modal attention (AIDO 1280-dim query over STRING keys) caused
     regression due to 8.75:1 compression ratio and cross-space alignment difficulty
   - Intra-STRING attention (STRING center as query, STRING neighbors as keys/values)
     was the design used in node2-2-1-1 (F1=0.5110) and node2-1-1-1-1-1 (F1=0.5128)
   - Both query and keys live in the same 256-dim STRING embedding space (2:1 ratio only)

2. Revert LoRA r=16 -> r=8:
   - r=16 without adjusted LR caused under-optimization in the parent
   - r=8 consistently reaches 0.51+ F1 in the best lineages of this task
   - Fewer backbone trainable params means faster, more stable convergence

3. Revert head_hidden=512 -> 256:
   - 512-dim head has ~12M params for only 1,536 input features and 1,388 train samples
   - 256-dim head (~5M params) has proven better generalization on this small dataset
   - With only 1,388 training samples, reducing head capacity helps prevent overfitting

4. Discriminative learning rates (primary new innovation):
   - backbone_lr=5e-5 for LoRA parameters (lower, stable adaptation)
   - head_lr=1.5e-4 for head + neighborhood attention (higher, faster learning)
   - This matches the DLR approach from node2-1-1 lineage (best: F1=0.4921, +0.025
     over non-DLR parent) and node2-1-1-1-1-1 (F1=0.5128 tree best uses similar config)
   - Single global lr=8e-5 was inherited from parent but may be suboptimal with r=8

5. Extended training: max_epochs=250, patience=25
   - Parent used max_epochs=200, patience=20 (may have cut training short)
   - Previous node2-2 lineage showed slow improvement in late epochs warranting more patience
   - Provides more opportunity for the model to find better minima after warmup

6. Keep FUSION_DIM=1536 (1280 AIDO + 256 STRING):
   - Uses concat(pert_gene_emb [B, 640], summary_token [B, 640]) = [B, 1280]
   - The dual AIDO feature (pert + summary) has shown value in the node2-2 lineage
   - Intra-STRING attention produces [B, 256]
   - Fusion: [B, 1536] feeds MLP head

Architecture:
    AIDO.Cell-100M (bf16, gradient checkpointing)
      + LoRA(r=8, alpha=16, dropout=0.05) on Q/K/V x 18 layers
    Features: concat(pert_gene_emb, summary_emb) -> [B, 1280] (float32)

    STRING_GNN (frozen, pre-computed embeddings)
      NeighborhoodAttentionModule (intra-STRING, 2-head, attn_dim=64):
        Query = STRING center embedding [B, 256]
        Keys/Values = STRING neighbor embeddings [B, K=16, 256]
    Features: concat(center_proj, context_attn) -> [B, 256]

    Fusion: concat([B, 1280], [B, 256]) -> [B, 1536]
    Head: Linear(1536->256) -> LN -> GELU -> Dropout(0.45) -> Linear(256->3*6640)
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
AIDO_GENES = 19264       # AIDO.Cell gene vocabulary size
AIDO_DIR   = "/home/Models/AIDO.Cell-100M"
STRING_DIR = "/home/Models/STRING_GNN"
HIDDEN_DIM = 640         # AIDO.Cell-100M hidden size
STRING_DIM = 256         # STRING_GNN output dimension
FUSION_DIM = HIDDEN_DIM * 2 + STRING_DIM  # 1280 + 256 = 1536

# Class frequencies: [down(-1->0), neutral(0->1), up(+1->2)]
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency class weights for weighted CE."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py.

    Args:
        preds:   [N, 3, G] softmax probabilities
        targets: [N, G]    integer class labels in {0,1,2}

    Returns:
        scalar float F1
    """
    y_hat = preds.argmax(dim=1)   # [N, G]
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
# STRING_GNN Neighborhood Attention Module (intra-STRING: center queries neighbors)
# ---------------------------------------------------------------------------
class NeighborhoodAttentionModule(nn.Module):
    """Aggregate top-K PPI neighbors of the perturbed gene via multi-head attention.

    For each perturbed gene, retrieves its top-K STRING neighbors (by edge weight)
    and computes 2-head attention (STRING center as query, STRING neighbors as keys/values)
    to produce a context embedding concatenated with the center node embedding.

    This is the INTRA-STRING attention design (not cross-modal), which is the
    approach that achieved F1=0.5110 and F1=0.5128 in the node2 lineage.
    The query and keys/values live in the same 256-dim STRING embedding space,
    making the 2:1 compression ratio benign vs. the cross-modal 8.75:1 ratio.

    Output: [B, STRING_DIM] where STRING_DIM=256 (center=128, context=128)
    """
    def __init__(
        self,
        k: int = 16,
        n_heads: int = 2,
        attn_dim: int = 64,
        dropout: float = 0.1,
        string_dim: int = 256,
    ) -> None:
        super().__init__()
        self.k = k
        self.n_heads = n_heads
        self.attn_dim = attn_dim
        self.string_dim = string_dim
        total_attn_dim = n_heads * attn_dim  # 2*64=128

        # Center and context projections (each half of STRING_DIM)
        self.center_proj  = nn.Linear(string_dim, string_dim // 2)    # 256 -> 128
        self.context_proj = nn.Linear(string_dim, string_dim // 2)    # 256 -> 128

        # Multi-head attention for neighbor aggregation (query=center, k/v=neighbors)
        self.q_proj   = nn.Linear(string_dim, total_attn_dim)         # 256 -> 128
        self.k_proj   = nn.Linear(string_dim, total_attn_dim)         # 256 -> 128
        self.v_proj   = nn.Linear(string_dim, total_attn_dim)         # 256 -> 128
        self.out_proj = nn.Linear(total_attn_dim, string_dim // 2)    # 128 -> 128

        self.norm    = nn.LayerNorm(string_dim // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        center_embs: torch.Tensor,      # [B, 256]
        neighbor_embs: torch.Tensor,    # [B, K, 256]
        neighbor_weights: torch.Tensor, # [B, K] normalized weights
    ) -> torch.Tensor:                  # [B, 256]
        B, K, D = neighbor_embs.shape

        # Center branch: direct projection of the center node embedding
        center_out = self.center_proj(center_embs)   # [B, 128]

        # Context branch: multi-head attention (query = center, k/v = neighbors)
        q   = self.q_proj(center_embs).unsqueeze(1)  # [B, 1, 128]
        k_n = self.k_proj(neighbor_embs)              # [B, K, 128]
        v_n = self.v_proj(neighbor_embs)              # [B, K, 128]

        # Reshape for multi-head attention
        q   = q.view(B, 1, self.n_heads, self.attn_dim).transpose(1, 2)    # [B, H, 1, d]
        k_n = k_n.view(B, K, self.n_heads, self.attn_dim).transpose(1, 2)  # [B, H, K, d]
        v_n = v_n.view(B, K, self.n_heads, self.attn_dim).transpose(1, 2)  # [B, H, K, d]

        # Scaled dot-product attention + STRING edge weight bias
        scale  = self.attn_dim ** -0.5
        scores = torch.matmul(q, k_n.transpose(-2, -1)) * scale  # [B, H, 1, K]
        weight_bias = neighbor_weights.unsqueeze(1).unsqueeze(2).log().clamp(min=-10)  # [B, 1, 1, K]
        scores = scores + weight_bias

        attn    = torch.softmax(scores, dim=-1)           # [B, H, 1, K]
        attn    = self.dropout(attn)

        context = torch.matmul(attn, v_n)                 # [B, H, 1, d]
        context = context.squeeze(2)                       # [B, H, d]
        context = context.transpose(1, 2).contiguous().view(B, -1)  # [B, H*d=128]
        context = self.out_proj(context)                   # [B, 128]
        context = self.norm(context)

        # Concatenate center (STRING identity) and context (neighbor-attention output)
        return torch.cat([center_out, context], dim=-1)  # [B, 256]


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
        tokenized = tokenizer(expr_dicts, return_tensors="pt")

        # Find perturbed gene position for each sample (position where value != -1.0)
        input_ids = tokenized["input_ids"]   # [B, 19264]
        gene_in_vocab  = (input_ids > -1.0).any(dim=1)
        gene_positions = torch.where(
            gene_in_vocab,
            (input_ids > -1.0).float().argmax(dim=1),
            torch.zeros(len(batch), dtype=torch.long),
        )  # [B]

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
class AIDOCellStringFusionModel(pl.LightningModule):
    """AIDO.Cell-100M LoRA(r=8) + Intra-STRING Neighborhood Attention + Discriminative LRs.

    Architecture:
        AIDO.Cell-100M (bf16, gradient checkpointing)
          LoRA(r=8, alpha=16) on Q, K, V across all 18 layers
          Features: concat(pert_gene_emb, summary_token) -> [B, 1280] (float32)

        STRING_GNN (frozen, cached pre-run)
          NeighborhoodAttentionModule (intra-STRING 2-head):
            Query = STRING center embedding [B, 256]
            Keys/Values = STRING neighbor embeddings [B, K=16, 256]
          Features: concat(center_proj, context_attn) -> [B, 256] (float32)

        Fusion: concat([B, 1280], [B, 256]) -> [B, 1536]
        Head: Linear(1536->256) -> LN -> GELU -> Dropout(0.45) -> Linear(256->3*6640)

    Loss: Weighted CE + label smoothing epsilon=0.05
    Optimizer: AdamW with discriminative LRs (backbone_lr=5e-5, head_lr=1.5e-4)
    """

    def __init__(
        self,
        lora_r: int            = 8,
        lora_alpha: int        = 16,
        lora_dropout: float    = 0.05,
        head_hidden: int       = 256,
        head_dropout: float    = 0.45,
        backbone_lr: float     = 5e-5,
        head_lr: float         = 1.5e-4,
        weight_decay: float    = 2e-2,
        warmup_epochs: int     = 10,
        label_smoothing: float = 0.05,
        nb_k: int              = 16,
        nb_heads: int          = 2,
        nb_attn_dim: int       = 64,
        max_epochs: int        = 250,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Accumulators for val/test
        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # -------- AIDO.Cell-100M backbone --------
        backbone = AutoModel.from_pretrained(AIDO_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Monkey-patch to avoid NotImplementedError for custom GeneEmbedding
        backbone.enable_input_require_grads = lambda: None

        # Wrap with LoRA
        lora_cfg = LoraConfig(
            task_type      = TaskType.FEATURE_EXTRACTION,
            r              = hp.lora_r,
            lora_alpha     = hp.lora_alpha,
            lora_dropout   = hp.lora_dropout,
            target_modules = ["query", "key", "value"],
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Register forward hook on gene_embedding to enable gradient flow
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        gene_emb = self.backbone.model.bert.gene_embedding
        gene_emb.register_forward_hook(_make_inputs_require_grad)

        # Cast LoRA params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # -------- STRING_GNN: load + freeze + cache embeddings --------
        string_model_dir = Path(STRING_DIR)
        string_gnn = AutoModel.from_pretrained(str(string_model_dir), trust_remote_code=True)
        string_gnn = string_gnn.cuda()
        string_gnn.eval()

        graph = torch.load(string_model_dir / "graph_data.pt")
        node_names = json.loads((string_model_dir / "node_names.json").read_text())

        # Build Ensembl ID -> node index mapping
        self._ens2idx: Dict[str, int] = {name: i for i, name in enumerate(node_names)}

        edge_index  = graph["edge_index"].cuda()
        edge_weight = graph["edge_weight"]
        if edge_weight is not None:
            edge_weight = edge_weight.cuda()

        # Pre-compute STRING embeddings for all nodes (frozen once)
        with torch.no_grad():
            string_out = string_gnn(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            # Cache all node embeddings as a buffer [18870, 256]
            self.register_buffer("_string_embs", string_out.last_hidden_state.float().cpu())

        # Pre-build neighbor index for each STRING node (top-K by edge weight)
        n_nodes = len(node_names)
        E = edge_index.shape[1]

        # Build adjacency: src -> list of (dst, weight)
        adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n_nodes)}
        edge_index_np = edge_index.cpu().numpy()
        if edge_weight is not None:
            edge_weight_np = edge_weight.cpu().numpy()
        else:
            edge_weight_np = np.ones(E, dtype=np.float32)

        for e_idx in range(E):
            src = int(edge_index_np[0, e_idx])
            dst = int(edge_index_np[1, e_idx])
            w   = float(edge_weight_np[e_idx])
            adj[src].append((dst, w))

        # For each node, sort neighbors by weight desc and take top-K
        K = hp.nb_k
        self._nb_idx = torch.zeros(n_nodes, K, dtype=torch.long)     # [N, K]
        self._nb_wts = torch.zeros(n_nodes, K, dtype=torch.float32)  # [N, K]

        for node_i in range(n_nodes):
            nbrs = adj[node_i]
            if len(nbrs) == 0:
                # No neighbors: self-loop
                self._nb_idx[node_i] = node_i
                self._nb_wts[node_i] = 1.0
            else:
                # Sort by weight descending
                nbrs_sorted = sorted(nbrs, key=lambda x: x[1], reverse=True)[:K]
                actual_k = len(nbrs_sorted)
                for j, (dst, w) in enumerate(nbrs_sorted):
                    self._nb_idx[node_i, j] = dst
                    self._nb_wts[node_i, j] = w
                # Pad with self-loops if fewer than K neighbors
                for j in range(actual_k, K):
                    self._nb_idx[node_i, j] = node_i
                    self._nb_wts[node_i, j] = 0.0
                # Normalize weights (softmax-style)
                wts = self._nb_wts[node_i, :actual_k].clone()
                wts = wts / (wts.sum() + 1e-8)
                self._nb_wts[node_i, :actual_k] = wts

        # Register as buffers so they move with the model
        self.register_buffer("string_nb_idx", self._nb_idx)
        self.register_buffer("string_nb_wts", self._nb_wts)

        # -------- Intra-STRING neighborhood attention module --------
        self.nb_attn = NeighborhoodAttentionModule(
            k          = hp.nb_k,
            n_heads    = hp.nb_heads,
            attn_dim   = hp.nb_attn_dim,
            dropout    = 0.1,
            string_dim = STRING_DIM,
        )

        # -------- Classification head --------
        # Input: 1280 (AIDO concat: pert_emb + summary_emb) + 256 (STRING) = 1536
        self.head = nn.Sequential(
            nn.Linear(FUSION_DIM, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )

        # Class weights for weighted CE
        self.register_buffer("class_weights", get_class_weights())

    # -------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------
    def _get_string_features(
        self,
        pert_ids: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute intra-STRING neighborhood-attention-aggregated features.

        Args:
            pert_ids: list of Ensembl IDs for the perturbed genes
            device:   target device

        Returns:
            [B, 256] STRING neighborhood embedding
        """
        B = len(pert_ids)
        string_embs = self._string_embs.to(device)  # [N, 256]

        # Map pert_ids to STRING node indices (fallback to 0 if not found)
        node_indices = torch.tensor(
            [self._ens2idx.get(pid, 0) for pid in pert_ids],
            dtype=torch.long, device=device,
        )  # [B]

        # Center embeddings
        center_embs = string_embs[node_indices]  # [B, 256]

        # Neighbor indices and weights
        nb_idx = self.string_nb_idx[node_indices]  # [B, K]
        nb_wts = self.string_nb_wts[node_indices]  # [B, K]

        # Neighbor embeddings
        nb_embs = string_embs[nb_idx.view(-1)].view(B, self.hparams.nb_k, STRING_DIM)  # [B, K, 256]

        # Normalize weights to sum to 1 per sample
        nb_wts_norm = nb_wts / (nb_wts.sum(dim=-1, keepdim=True) + 1e-8)  # [B, K]

        # Intra-STRING: center embedding queries its neighbors
        return self.nb_attn(center_embs, nb_embs, nb_wts_norm)  # [B, 256]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_positions: torch.Tensor,
        pert_ids: List[str],
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        device = input_ids.device

        # --- AIDO.Cell forward ---
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        lhs = out.last_hidden_state  # [B, 19266, 640]

        # 1. Perturbed gene embedding
        pert_emb = lhs[torch.arange(B, device=device), gene_positions, :]  # [B, 640]

        # 2. Summary token (position 19264 attends to all genes via full attention)
        summary_emb = lhs[:, AIDO_GENES, :]  # [B, 640]

        # Concatenate and cast to float32 for head
        aido_emb = torch.cat([pert_emb, summary_emb], dim=-1).float()  # [B, 1280]

        # --- STRING_GNN intra-STRING neighborhood attention ---
        string_emb = self._get_string_features(pert_ids, device)  # [B, 256]

        # --- Fusion ---
        fused = torch.cat([aido_emb, string_emb], dim=-1)  # [B, 1536]

        # --- Head ---
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)  # [B, 3, G]

        return logits

    # -------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy with label smoothing."""
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),
            targets.reshape(-1),
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # -------------------------------------------------------------------
    # Training / Validation / Test steps
    # -------------------------------------------------------------------
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

        # All-gather across GPUs
        all_preds = self.all_gather(local_preds.cuda()).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts.cuda()).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx.cuda()).view(-1)

        # Sort and deduplicate
        order   = torch.argsort(all_idx)
        s_idx   = all_idx[order]
        s_pred  = all_preds[order]
        s_tgt   = all_tgts[order]
        mask = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])

        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val_f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["pert_id"],
        )
        probs = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs.cpu())
        self._test_idx.append(batch["sample_idx"].detach().cpu())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)
        local_idx   = torch.cat(self._test_idx,   0)
        self._test_preds.clear(); self._test_idx.clear()

        # All-gather across GPUs
        all_preds = self.all_gather(local_preds.cuda()).view(-1, N_CLASSES, N_GENES)
        all_idx   = self.all_gather(local_idx.cuda()).view(-1)

        if self.trainer.is_global_zero:
            # Sort and deduplicate: both tensors must be masked together
            order   = torch.argsort(all_idx)
            s_idx   = all_idx[order]
            s_pred  = all_preds[order]
            mask = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
            # Apply mask to both tensors to keep them aligned
            dedup_idx   = s_idx[mask]    # shape: (M,)
            dedup_pred  = s_pred[mask]   # shape: (M, 3, 6640)

            # Store predictions (for potential external use)
            self._final_test_preds = dedup_pred
            self._final_test_idx   = dedup_idx

            # Retrieve pert_id and symbol from dataset using integer indices
            test_ds = self.trainer.datamodule.test_ds
            rows = []
            for i in range(dedup_idx.shape[0]):
                sample_i = int(dedup_idx[i].item())
                pid = test_ds.pert_ids[sample_i]
                sym = test_ds.symbols[sample_i]
                rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": json.dumps(dedup_pred[i].float().cpu().numpy().tolist()),
                })

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node2-2-1-1-1-1] Saved {len(rows)} test predictions.")

    # -------------------------------------------------------------------
    # Checkpoint: save only trainable params + neighborhood attention + buffers
    # -------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                key = prefix + name
                if key in full:
                    trainable[key] = full[key]
        # Also save neighborhood attention module (always trainable)
        for name, p in self.nb_attn.named_parameters():
            key = prefix + "nb_attn." + name
            if key in full:
                trainable[key] = full[key]
        # Save head parameters
        for name, p in self.head.named_parameters():
            key = prefix + "head." + name
            if key in full:
                trainable[key] = full[key]
        # Always save buffers
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

    # -------------------------------------------------------------------
    # Optimizer: Discriminative LRs
    # -------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams

        # Separate parameter groups: LoRA backbone (lower LR) vs head+attn (higher LR)
        backbone_params = [p for n, p in self.backbone.named_parameters() if p.requires_grad]
        head_params     = list(self.head.parameters()) + list(self.nb_attn.parameters())

        param_groups = [
            {"params": backbone_params, "lr": hp.backbone_lr, "weight_decay": hp.weight_decay},
            {"params": head_params,     "lr": hp.head_lr,     "weight_decay": hp.weight_decay},
        ]
        opt = torch.optim.AdamW(param_groups)

        # Linear warmup -> cosine annealing
        total_epochs  = hp.max_epochs
        warmup_epochs = hp.warmup_epochs

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return max(1e-7 / max(hp.backbone_lr, hp.head_lr),
                       0.5 * (1 + np.cos(np.pi * progress)))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node2-2-1-1-1-1: AIDO.Cell-100M LoRA(r=8) + Intra-STRING Attention "
                    "+ Discriminative LRs (backbone_lr=5e-5, head_lr=1.5e-4)"
    )
    parser.add_argument("--micro-batch-size",   type=int,   default=4)
    parser.add_argument("--global-batch-size",  type=int,   default=32)
    parser.add_argument("--max-epochs",         type=int,   default=250)
    parser.add_argument("--backbone-lr",        type=float, default=5e-5)
    parser.add_argument("--head-lr",            type=float, default=1.5e-4)
    parser.add_argument("--weight-decay",       type=float, default=2e-2)
    parser.add_argument("--lora-r",             type=int,   default=8)
    parser.add_argument("--lora-alpha",         type=int,   default=16)
    parser.add_argument("--lora-dropout",       type=float, default=0.05)
    parser.add_argument("--head-hidden",        type=int,   default=256)
    parser.add_argument("--head-dropout",       type=float, default=0.45)
    parser.add_argument("--warmup-epochs",      type=int,   default=10)
    parser.add_argument("--label-smoothing",    type=float, default=0.05)
    parser.add_argument("--nb-k",               type=int,   default=16)
    parser.add_argument("--nb-heads",           type=int,   default=2)
    parser.add_argument("--nb-attn-dim",        type=int,   default=64)
    parser.add_argument("--num-workers",        type=int,   default=4)
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--patience",           type=int,   default=25)
    parser.add_argument("--debug_max_step",     type=int,   default=None)
    parser.add_argument("--fast_dev_run",       action="store_true")
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
    model = AIDOCellStringFusionModel(
        lora_r          = args.lora_r,
        lora_alpha      = args.lora_alpha,
        lora_dropout    = args.lora_dropout,
        head_hidden     = args.head_hidden,
        head_dropout    = args.head_dropout,
        backbone_lr     = args.backbone_lr,
        head_lr         = args.head_lr,
        weight_decay    = args.weight_decay,
        warmup_epochs   = args.warmup_epochs,
        label_smoothing = args.label_smoothing,
        nb_k            = args.nb_k,
        nb_heads        = args.nb_heads,
        nb_attn_dim     = args.nb_attn_dim,
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
    es_cb = EarlyStopping(
        monitor   = "val_f1",
        mode      = "max",
        patience  = args.patience,
        min_delta = 1e-3,
    )
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

    # Run test with the best checkpoint
    if fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=dm)
    else:
        test_results = trainer.test(model, datamodule=dm, ckpt_path="best")

    # Only rank 0 writes test_score.txt to avoid race conditions in DDP
    if trainer.global_rank == 0:
        score_path = Path(__file__).parent / "test_score.txt"
        with open(score_path, "w") as f:
            f.write(f"test_results: {test_results}\n")
            if test_results:
                for k, v in test_results[0].items():
                    f.write(f"  {k}: {v}\n")
        print(f"[Node2-2-1-1-1-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
