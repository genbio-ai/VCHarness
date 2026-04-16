"""Node 2-1-2-1: AIDO.Cell-100M (LoRA r=8) + STRING_GNN K=16 2-head Neighborhood Attention Fusion
with Corrected Backbone LR and Extended Training.

Improvements over parent node2-1-2 (F1=0.4921):
1. **Backbone LR fix** (primary): backbone_lr 5e-5 → 1e-4
   - Parent's conservative 5e-5 was the primary performance bottleneck (4× ratio to head)
   - Tree evidence: node2-2 (unified 1e-4) → F1=0.5078; node2-1-1-1-1-1 (unified 1e-4) → F1=0.5128
   - GatedFusion was NOT applied — confirmed harmful for LoRA-adapted AIDO.Cell (node4-2-2-1-1: -0.032)
   - GenePriorBias was NOT applied — confirmed harmful for AIDO.Cell lineage
2. **Extended training**: max_epochs 200 → 300, patience 12 → 20
   - node2-1-1-1-1-1 peaked at epoch 77; patience=10 barely captured it
   - node2-2-1 peaked at epoch 107 with patience=20 → F1=0.5110
   - With 300 max_epochs and patience=20, late peaks can be captured reliably
3. **Reduced warmup**: warmup_epochs 10 → 5
   - head_lr=2e-4 allows rapid head convergence; 10-epoch warmup wastes ~18% of training
4. **Reduced head_dropout**: 0.5 → 0.45
   - node2-2-1 showed dropout 0.45 improves over 0.5 (+0.003 F1)
   - Provides slightly more head capacity without risking overfitting
5. **All proven regularization retained** (no arch changes vs. parent):
   - STRING K=16 2-head neighborhood attention (proven best)
   - Simple concat fusion (GatedFusion CONFIRMED HARMFUL for AIDO.Cell LoRA)
   - head_hidden=256, weight_decay=2e-2, gradient_clip=1.0
   - Weighted CE + label smoothing ε=0.05
   - CosineAnnealingLR with raised eta_min=1e-6
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES    = 6640
N_CLASSES  = 3
AIDO_GENES = 19264           # AIDO.Cell gene vocabulary size
MODEL_DIR  = "/home/Models/AIDO.Cell-100M"
STRING_DIR = "/home/Models/STRING_GNN"
HIDDEN_DIM = 640             # AIDO.Cell-100M hidden size
STRING_DIM = 256             # STRING_GNN output dim

# Class frequency: [down(-1→0), neutral(0→1), up(+1→2)]
CLASS_FREQ = [0.0429, 0.9251, 0.0320]
LABEL_SMOOTHING = 0.05

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
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds: [N, 3, G] float softmax probabilities
        targets: [N, G] long class labels in {0, 1, 2}
    Returns:
        scalar F1 score
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
        f1_c = torch.where(
            prec + rec > 0,
            2 * prec * rec / (prec + rec + 1e-8),
            torch.zeros_like(prec),
        )
        f1_per_gene += f1_c * present
        n_present   += present

    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# STRING_GNN Neighborhood Attention Module
# ---------------------------------------------------------------------------
class NeighborhoodAttentionModule(nn.Module):
    """K-nearest neighbor attention over PPI graph for perturbed gene context.

    For each perturbed gene:
    - Retrieves pre-computed STRING_GNN embeddings for its top-K PPI neighbors
    - Applies 2-head multi-head attention (query=center, keys/values=neighbors)
    - Adds STRING edge weight biases to attention scores for topology-aware weighting
    - Outputs concatenation of center projection (dim/2) + context projection (dim/2)

    Architecture proven in node2-1-1-1-1-1 (F1=0.5128) and node2-2 (F1=0.5078).
    Simple concat fusion is used (GatedFusion CONFIRMED harmful: node4-2-2-1-1 F1=0.4738).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 2,
        k_neighbors: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_heads   = num_heads
        self.k_neighbors = k_neighbors
        self.head_dim    = embed_dim // num_heads

        # Multi-head attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Center and context projections for output concatenation
        self.center_proj  = nn.Linear(embed_dim, embed_dim // 2, bias=False)
        self.context_proj = nn.Linear(embed_dim, embed_dim // 2, bias=False)

        # Output normalization
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        center_emb: torch.Tensor,       # [B, D]
        neighbor_embs: torch.Tensor,    # [B, K, D]
        edge_weights: torch.Tensor,     # [B, K] STRING edge confidence weights
    ) -> torch.Tensor:
        """Returns [B, D] fused STRING_GNN neighborhood representation."""
        B, K, D = neighbor_embs.shape
        H = self.num_heads
        head_dim = D // H
        scale = head_dim ** -0.5

        # Project center as query, neighbors as keys and values
        q = self.q_proj(center_emb).view(B, H, head_dim)        # [B, H, d]
        k = self.k_proj(neighbor_embs).view(B, K, H, head_dim)  # [B, K, H, d]
        v = self.v_proj(neighbor_embs).view(B, K, H, head_dim)  # [B, K, H, d]

        # Transpose for attention computation: [B, H, 1, d] x [B, H, d, K] = [B, H, 1, K]
        q = q.unsqueeze(2)    # [B, H, 1, d]
        k = k.permute(0, 2, 1, 3)  # [B, H, K, d]
        v = v.permute(0, 2, 1, 3)  # [B, H, K, d]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, 1, K]

        # Add STRING edge weight bias (topology-aware: higher-confidence edges get attention boost)
        # edge_weights: [B, K] → [B, 1, 1, K]
        attn_scores = attn_scores + edge_weights.unsqueeze(1).unsqueeze(2)

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, 1, K]
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values: [B, H, 1, K] x [B, H, K, d] = [B, H, 1, d]
        context = torch.matmul(attn_weights, v)        # [B, H, 1, d]
        context = context.squeeze(2).view(B, D).float()  # [B, D]

        center_emb = center_emb.float()
        # Apply layer norm for stability
        context = self.norm(context)

        # Concatenate center and context projections: [B, D/2] + [B, D/2] = [B, D]
        out = torch.cat([
            self.center_proj(center_emb),
            self.context_proj(context),
        ], dim=-1)

        return out  # [B, D]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
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

        # Tokenize: each sample gets only its perturbed gene with expression=1.0;
        # all other 19,263 genes receive -1.0 (missing).
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized = tokenizer(expr_dicts, return_tensors="pt")  # input_ids: [B, 19264] float32

        # Find the gene position for each sample (position where input_ids > -1.0)
        input_ids = tokenized["input_ids"]   # [B, 19264]
        gene_in_vocab  = (input_ids > -1.0).any(dim=1)                        # [B]
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
        self.tokenizer: Optional[Any] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Rank-0 downloads first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df)
        self.val_ds   = DEGDataset(val_df)
        self.test_ds  = DEGDataset(test_df)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_collate(self.tokenizer),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------
class AIDOCellStringFusionModel(pl.LightningModule):
    """AIDO.Cell-100M LoRA(r=8) + STRING_GNN K=16 2-head neighborhood attention fusion.

    Feature streams:
      1. AIDO.Cell-100M (LoRA fine-tuned): concat(pert_gene_emb [640], summary_token [640]) = [1280]
      2. STRING_GNN (frozen): K=16 2-head neighborhood attention → [256]

    Fusion: simple concat([1280, 256]) = [1536] → head → [B, 3, 6640]

    Key improvements over parent node2-1-2 (F1=0.4921):
      - backbone_lr: 5e-5 → 1e-4 (primary fix — discriminative 4× ratio was too conservative)
      - head_dropout: 0.5 → 0.45 (proven improvement: node2-2-1 showed +0.003)
      - max_epochs: 200 → 300 (extended training captures late peaks like epoch 77 in tree-best)
      - warmup_epochs: 10 → 5 (head converges quickly at head_lr=2e-4)
      - patience: 12 → 20 (reliably captures late peaks; parent wasted 13 post-peak epochs)

    Architecture kept identical to parent (no gated fusion — confirmed harmful for LoRA-AIDO.Cell).

    Tree context:
      - node2-2 (F1=0.5078): same arch, unified lr=1e-4 → validates backbone_lr=1e-4
      - node2-1-1-1-1-1 (F1=0.5128, tree best): same arch, unified lr=1e-4, patience=10
      - node4-2-2-1-1 (F1=0.4738): GatedFusion + AIDO LoRA → CONFIRMED HARMFUL
    """

    def __init__(
        self,
        lora_r: int         = 8,
        lora_alpha: int     = 16,
        lora_dropout: float = 0.05,
        head_hidden: int    = 256,
        head_dropout: float = 0.45,     # ← 0.5 → 0.45 (node2-2-1 showed improvement)
        backbone_lr: float  = 1e-4,     # ← 5e-5 → 1e-4 (primary fix: match proven best nodes)
        head_lr: float      = 2e-4,     # keep same: maintains 2× discriminative ratio
        weight_decay: float = 2e-2,
        warmup_epochs: int  = 5,        # ← 10 → 5 (head converges quickly)
        max_epochs: int     = 300,      # ← 200 → 300 (extend to capture late peaks)
        k_neighbors: int    = 16,
        attn_dropout: float = 0.1,
        string_eta_min: float = 1e-6,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load AIDO.Cell backbone ----
        backbone = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Monkey-patch enable_input_require_grads for AIDO.Cell compatibility
        backbone.enable_input_require_grads = lambda: None

        # Apply LoRA to Q/K/V across all 18 transformer layers
        lora_cfg = LoraConfig(
            task_type      = TaskType.FEATURE_EXTRACTION,
            r              = hp.lora_r,
            lora_alpha     = hp.lora_alpha,
            lora_dropout   = hp.lora_dropout,
            target_modules = ["query", "key", "value"],
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Forward hook to ensure gradients flow through gene_embedding
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        gene_emb = self.backbone.model.bert.gene_embedding
        gene_emb.register_forward_hook(_make_inputs_require_grad)

        # Cast LoRA params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ---- Load STRING_GNN and pre-compute graph embeddings ----
        string_model_dir = Path(STRING_DIR)
        string_model = AutoModel.from_pretrained(str(string_model_dir), trust_remote_code=True)
        string_model.eval()

        graph_data  = torch.load(string_model_dir / "graph_data.pt", map_location="cpu")
        node_names  = json.loads((string_model_dir / "node_names.json").read_text())

        # Build Ensembl ID → node index lookup
        self._string_node_idx: Dict[str, int] = {name: i for i, name in enumerate(node_names)}
        n_nodes = len(node_names)

        # Pre-compute all STRING_GNN node embeddings (frozen)
        with torch.no_grad():
            edge_index  = graph_data["edge_index"]
            edge_weight = graph_data.get("edge_weight", None)
            string_out  = string_model(edge_index=edge_index, edge_weight=edge_weight)
            string_embs = string_out.last_hidden_state.float().cpu()  # [18870, 256]

        # Register as buffer (non-trainable, moves with model)
        self.register_buffer("string_embs", string_embs)  # [N_nodes, 256]

        # Pre-compute top-K neighbor indices and edge weights for each node
        # edge_index: [2, E] — source → destination edges
        edge_index_np   = graph_data["edge_index"].numpy()
        edge_weight_np  = (
            graph_data["edge_weight"].numpy()
            if edge_weight is not None
            else np.ones(edge_index_np.shape[1], dtype=np.float32)
        )

        K = hp.k_neighbors
        # Build adjacency list: node → list of (neighbor_idx, weight)
        adjacency: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n_nodes)}
        for e in range(edge_index_np.shape[1]):
            src, dst = int(edge_index_np[0, e]), int(edge_index_np[1, e])
            adjacency[src].append((dst, float(edge_weight_np[e])))

        # Build fixed K neighbor lookup tensors
        neighbor_idx_list    = []
        neighbor_weight_list = []
        for node_i in range(n_nodes):
            nbrs = adjacency[node_i]
            # Sort by edge weight (desc) and take top-K
            nbrs_sorted = sorted(nbrs, key=lambda x: x[1], reverse=True)[:K]
            idxs    = [nb[0] for nb in nbrs_sorted]
            weights = [nb[1] for nb in nbrs_sorted]
            # Pad to K with self-loops and zero weights if fewer than K neighbors
            while len(idxs) < K:
                idxs.append(node_i)
                weights.append(0.0)
            neighbor_idx_list.append(idxs[:K])
            neighbor_weight_list.append(weights[:K])

        neighbor_idx_tensor    = torch.tensor(neighbor_idx_list,    dtype=torch.long)     # [N, K]
        neighbor_weight_tensor = torch.tensor(neighbor_weight_list, dtype=torch.float32)  # [N, K]
        self.register_buffer("neighbor_idx",    neighbor_idx_tensor)    # [N_nodes, K]
        self.register_buffer("neighbor_weights", neighbor_weight_tensor) # [N_nodes, K]

        # Number of nodes (for OOV fallback)
        self._n_string_nodes = n_nodes

        # ---- STRING_GNN neighborhood attention module ----
        self.string_attn = NeighborhoodAttentionModule(
            embed_dim   = STRING_DIM,
            num_heads   = 2,
            k_neighbors = K,
            dropout     = hp.attn_dropout,
        )
        # Cast to float32
        for param in self.string_attn.parameters():
            param.data = param.data.float()

        # ---- Classification head ----
        # Input: concat(aido_emb [2*640=1280], string_emb [256]) = 1536
        # Simple concat fusion — GatedFusion confirmed harmful for LoRA-AIDO.Cell (node4-2-2-1-1)
        in_dim = 2 * HIDDEN_DIM + STRING_DIM  # 1280 + 256 = 1536
        self.head = nn.Sequential(
            nn.Linear(in_dim, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),  # 0.45 (reduced from 0.5; node2-2-1 showed improvement)
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )

        # ---- Loss weights ----
        self.register_buffer("class_weights", get_class_weights())

        # ---- Accumulators ----
        self._val_preds: List[torch.Tensor] = []
        self._val_tgts:  List[torch.Tensor] = []
        self._val_idx:   List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    def _get_string_features(self, pert_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve center embeddings, neighbor embeddings, and edge weights for a batch.

        Returns:
            center_embs: [B, 256] — STRING embedding for each perturbed gene
            neighbor_embs: [B, K, 256] — top-K neighbor embeddings
            edge_weights: [B, K] — STRING edge confidence weights
        """
        device = self.string_embs.device
        B = len(pert_ids)
        K = self.hparams.k_neighbors

        # Map pert_ids to STRING node indices (with OOV fallback: use node 0)
        node_indices = torch.zeros(B, dtype=torch.long, device=device)
        for i, pid in enumerate(pert_ids):
            # Strip version suffix if present (e.g., ENSG00000121410.12 → ENSG00000121410)
            pid_clean = pid.split(".")[0]
            node_indices[i] = self._string_node_idx.get(pid_clean, 0)

        # Center embeddings: [B, 256]
        center_embs = self.string_embs[node_indices]

        # Neighbor indices and weights: [B, K]
        nbr_idx     = self.neighbor_idx[node_indices]      # [B, K]
        nbr_weights = self.neighbor_weights[node_indices]  # [B, K]

        # Neighbor embeddings: [B, K, 256]
        neighbor_embs = self.string_embs[nbr_idx.view(-1)].view(B, K, STRING_DIM)

        return center_embs.float(), neighbor_embs.float(), nbr_weights.float()

    # ---- Forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_positions: torch.Tensor,
        pert_ids: List[str],
    ) -> torch.Tensor:
        B = input_ids.shape[0]

        # AIDO.Cell forward
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        # last_hidden_state: [B, 19266, 640]
        lhs = out.last_hidden_state

        # 1. Perturbed gene embedding
        pert_emb = lhs[torch.arange(B, device=lhs.device), gene_positions, :].float()  # [B, 640]

        # 2. Summary token (position 19264) — global transcriptome context
        summary_emb = lhs[:, AIDO_GENES, :].float()  # [B, 640]

        # AIDO stream: [B, 1280]
        aido_emb = torch.cat([pert_emb, summary_emb], dim=-1)

        # STRING_GNN stream: [B, 256]
        center_embs, neighbor_embs, edge_weights = self._get_string_features(pert_ids)
        string_emb = self.string_attn(center_embs, neighbor_embs, edge_weights)  # [B, 256]

        # Simple concat fusion: [B, 1536]
        # NOTE: GatedFusion is NOT used — confirmed harmful for LoRA-AIDO.Cell (node4-2-2-1-1)
        fused = torch.cat([aido_emb, string_emb], dim=-1)

        # Classification head: [B, 3, 6640]
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        return logits

    # ---- Loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),   # [B*G, 3]
            targets.reshape(-1),                        # [B*G]
            weight=self.class_weights,
            label_smoothing=LABEL_SMOOTHING,
        )

    # ---- Training step ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["gene_positions"],
            batch["pert_id"],
        )
        loss = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    # ---- Validation step ----
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["gene_positions"],
            batch["pert_id"],
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

        # Gather across all GPUs
        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        # Sort and deduplicate (handles DDP duplicates from DistributedSampler padding)
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

    # ---- Test step ----
    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["gene_positions"],
            batch["pert_id"],
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

        # Gather from all GPUs
        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        if self.trainer.is_global_zero:
            # Sort and deduplicate by sample index
            order  = torch.argsort(all_idx)
            s_idx  = all_idx[order]
            s_pred = all_preds[order]
            mask   = torch.cat([
                torch.tensor([True], device=s_idx.device),
                s_idx[1:] != s_idx[:-1],
            ])
            s_idx  = s_idx[mask]
            s_pred = s_pred[mask]

            # Retrieve pert_id / symbol from DataModule's test dataset
            test_ds = self.trainer.datamodule.test_ds

            rows = []
            for i in range(len(s_idx)):
                idx = s_idx[i].item()
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
            self.print(f"[node2-1-2-1] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()

    # ---- Checkpoint: save only trainable params ----
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_sd = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                key = prefix + name
                if key in full_sd:
                    trainable_sd[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                trainable_sd[key] = full_sd[key]
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Checkpoint: {trained}/{total} params ({100 * trained / total:.2f}%), "
            f"plus {buffers} buffer values"
        )
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer + scheduler (discriminative LR) ----
    def configure_optimizers(self):
        hp = self.hparams

        # Separate parameter groups for discriminative learning rates
        # backbone_lr=1e-4 (corrected from parent's 5e-5), head_lr=2e-4 (2× ratio)
        backbone_params = [p for n, p in self.backbone.named_parameters() if p.requires_grad]
        string_attn_params = list(self.string_attn.parameters())
        head_params = list(self.head.parameters())

        # backbone_lr raised to 1e-4 (matching proven best nodes node2-2, node2-1-1-1-1-1)
        # head_lr=2e-4: keeps 2× discriminative ratio (vs parent's 4× which was too conservative)
        param_groups = [
            {"params": backbone_params,     "lr": hp.backbone_lr,  "name": "backbone"},
            {"params": string_attn_params,  "lr": hp.head_lr,      "name": "string_attn"},
            {"params": head_params,         "lr": hp.head_lr,      "name": "head"},
        ]

        opt = torch.optim.AdamW(
            param_groups,
            weight_decay=hp.weight_decay,
        )

        warmup_epochs = hp.warmup_epochs
        total_epochs  = hp.max_epochs

        # 5-epoch linear warmup (reduced from 10 — head converges quickly at head_lr=2e-4)
        warmup_sched = LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        # Cosine annealing over 295 epochs (300-5) with eta_min=1e-6 floor
        cosine_sched = CosineAnnealingLR(
            opt,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=hp.string_eta_min,
        )
        scheduler = SequentialLR(
            opt,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
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
        description="Node 2-1-2-1: AIDO.Cell LoRA + STRING_GNN K=16 2-head fusion | Corrected backbone LR"
    )
    parser.add_argument("--micro-batch-size",   type=int,   default=4)
    parser.add_argument("--global-batch-size",  type=int,   default=32)
    parser.add_argument("--max-epochs",         type=int,   default=300)   # ← 200 → 300
    parser.add_argument("--backbone-lr",        type=float, default=1e-4,  # ← 5e-5 → 1e-4
                        dest="backbone_lr", help="LR for AIDO.Cell LoRA backbone")
    parser.add_argument("--head-lr",            type=float, default=2e-4,
                        dest="head_lr", help="LR for STRING attention + classification head")
    parser.add_argument("--weight-decay",       type=float, default=2e-2)
    parser.add_argument("--lora-r",             type=int,   default=8)
    parser.add_argument("--lora-alpha",         type=int,   default=16)
    parser.add_argument("--lora-dropout",       type=float, default=0.05)
    parser.add_argument("--head-hidden",        type=int,   default=256)
    parser.add_argument("--head-dropout",       type=float, default=0.45)  # ← 0.5 → 0.45
    parser.add_argument("--k-neighbors",        type=int,   default=16)
    parser.add_argument("--attn-dropout",       type=float, default=0.1)
    parser.add_argument("--warmup-epochs",      type=int,   default=5)     # ← 10 → 5
    parser.add_argument("--eta-min",            type=float, default=1e-6,
                        dest="string_eta_min")
    parser.add_argument("--patience",           type=int,   default=20)    # ← 12 → 20
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
        lim_train = lim_val = lim_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = lim_val = lim_test = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm = DEGDataModule(
        batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model = AIDOCellStringFusionModel(
        lora_r        = args.lora_r,
        lora_alpha    = args.lora_alpha,
        lora_dropout  = args.lora_dropout,
        head_hidden   = args.head_hidden,
        head_dropout  = args.head_dropout,
        backbone_lr   = args.backbone_lr,
        head_lr       = args.head_lr,
        weight_decay  = args.weight_decay,
        warmup_epochs = args.warmup_epochs,
        max_epochs    = args.max_epochs,
        k_neighbors   = args.k_neighbors,
        attn_dropout  = args.attn_dropout,
        string_eta_min = args.string_eta_min,
    )

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
        patience  = args.patience,  # 20 (increased from 12 to capture late peaks like epoch 77)
        min_delta = 1e-3,           # Fixed: matches F1 oscillation scale
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
    print(f"[node2-1-2-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
