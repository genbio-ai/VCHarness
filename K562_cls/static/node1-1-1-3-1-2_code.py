"""Node 1-2: AIDO.Cell-100M LoRA (r=16) + STRING_GNN K=16 2-Head + Focal Loss (NO GenePriorBias).

Improves on parent node1-1-1-3-1 (test F1=0.4610) and learns from sibling node1-1-1-3-1-1
(test F1=0.3669) by implementing the proven AIDO.Cell-100M + STRING_GNN fusion architecture
with critical improvements:

CORE STRATEGY: Fork from node2-1-1-1-1-1's proven architecture (AIDO.Cell-100M LoRA r=8 +
STRING_GNN K=16 2-head, test F1=0.5128) with THREE key innovations:

1. FOCAL LOSS (γ=1.5): Replace weighted CE with focal loss to dynamically focus on hard
   minority examples (DEG up/down classes, 7.5% of data). The per-gene macro F1 metric
   heavily rewards correct minority-class predictions — focal loss provides this focus
   without the catastrophic activation shock of GenePriorBias.

2. LoRA r=16 (vs proven r=8): Increased LoRA rank provides 2× more expressive adaptation
   parameters (~1.1M trainable vs ~553K), beneficial for learning diverse perturbation
   patterns across 1,388 distinct gene knockouts.

3. global_batch_size=256 (fix from sibling): The sibling (node1-1-1-3-1-1) used gbs=128,
   contributing to the worst generalization gap in the tree (val→test -0.0745). Node2
   lineage used gbs=256 and achieved zero gap. This fix is critical.

KEY LESSON: GenePriorBias is comprehensively proven harmful in EVERY context tested:
  - STRING_GNN lineage: identical -16% val F1 shock at activation (node1-1-1-3, node1-1-1-3-1)
  - AIDO.Cell fusion: catastrophic 16% drop causing worst generalization gap in tree (sibling)
  GenePriorBias is NOT included in this node.

Architecture:
    Stream 1 — AIDO.Cell-100M (LoRA r=16, α=32, ~1.1M trainable params):
        tokenize: {gene_ids: [pert_id], expression: [1.0]}
        → BERT-style transformer with LoRA on Q/K/V
        → last_hidden_state[:, 19264, :] = summary token [B, 640]

    Stream 2 — STRING_GNN K=16 2-Head Neighborhood Attention (frozen, 256-dim):
        pert_id → Ensembl lookup → STRING node embedding
        → 2-head attention over K=16 neighbors → [B, 256]

    Fusion: concat([aido_token, string_out]) → [B, 896]

    Head: Linear(896→256) → LayerNorm → GELU → Dropout(0.5) → Linear(256→19920)
         → view([B, 3, 6640])

    Loss: Focal loss (γ=1.5) with class weights (sqrt-inverse-frequency)
         → focuses training on hard DEG minority examples without disruption

Training: AdamW(lr=1e-4, weight_decay=2e-2), warmup(10 epochs) + CosineAnnealingLR(T_max=200),
          patience=20, max_epochs=300, global_batch_size=256
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from functools import partial
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
N_GENES = 6640
N_CLASSES = 3

# Remapped class frequencies (after -1→0, 0→1, 1→2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

AIDO_CELL_DIR = Path("/home/Models/AIDO.Cell-100M")
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

AIDO_DIM = 640   # AIDO.Cell-100M hidden size
STRING_DIM = 256  # STRING_GNN hidden dimension
FUSION_DIM = AIDO_DIM + STRING_DIM  # 896


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights; neutral class stays ~1."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def load_string_gnn_mapping() -> Dict[str, int]:
    """Load STRING_GNN node_names.json → Ensembl-ID to node-index mapping."""
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    return {name: idx for idx, name in enumerate(node_names)}


def precompute_neighborhood(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    n_nodes: int,
    K: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute top-K PPI neighbors for each node by edge confidence.

    Returns:
        neighbor_indices: [n_nodes, K] long — top-K neighbor node indices
                          (padded with -1 if fewer than K neighbors exist)
        neighbor_weights: [n_nodes, K] float — STRING confidence scores
    """
    src = edge_index[0]
    dst = edge_index[1]
    wgt = edge_weight

    sort_by_weight = torch.argsort(wgt, descending=True)
    src_sorted = src[sort_by_weight]
    dst_sorted = dst[sort_by_weight]
    wgt_sorted = wgt[sort_by_weight]

    sort_by_src = torch.argsort(src_sorted, stable=True)
    src_final = src_sorted[sort_by_src]
    dst_final = dst_sorted[sort_by_src]
    wgt_final = wgt_sorted[sort_by_src]

    counts = torch.bincount(src_final, minlength=n_nodes)

    neighbor_indices = torch.full((n_nodes, K), -1, dtype=torch.long)
    neighbor_weights = torch.zeros(n_nodes, K, dtype=torch.float32)

    start = 0
    for node_i in range(n_nodes):
        c = int(counts[node_i].item())
        if c == 0:
            start += c
            continue
        n_k = min(K, c)
        neighbor_indices[node_i, :n_k] = dst_final[start:start + n_k]
        neighbor_weights[node_i, :n_k] = wgt_final[start:start + n_k]
        start += c

    return neighbor_indices, neighbor_weights


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0, 1, 2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)  # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)
        is_pred = (y_hat == c)
        present = is_true.any(dim=0)

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
            "sample_idx":      idx,
            "pert_id":         self.pert_ids[idx],
            "symbol":          self.symbols[idx],
            "string_node_idx": self.string_node_indices[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]   # [G] in {0, 1, 2}
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_idx":      torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
        "pert_id":         [b["pert_id"]  for b in batch],
        "symbol":          [b["symbol"]   for b in batch],
        "string_node_idx": torch.stack([b["string_node_idx"] for b in batch]),
    }
    if "labels" in batch[0]:
        out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.string_map: Optional[Dict[str, int]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.string_map is None:
            self.string_map = load_string_gnn_mapping()

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df, self.string_map)
        self.val_ds   = DEGDataset(val_df,   self.string_map)
        self.test_ds  = DEGDataset(test_df,  self.string_map)

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
# Multi-Head Neighborhood Attention (2-head, proven from node2-1-1-1-1-1)
# ---------------------------------------------------------------------------
class MultiHeadNeighborhoodAttention(nn.Module):
    """2-head center-context gated attention over top-K PPI neighbors.

    Proven innovation from node2-1-1-1-1-1 (F1=0.5128):
    - 2 independent attention heads capture complementary neighborhood contexts
    - Head 1 may capture strong physical interactors
    - Head 2 may capture functional co-expression partners
    - Outputs concatenated [B, n_heads*embed_dim] then projected back to [B, embed_dim]
    """

    def __init__(self, embed_dim: int = 256, n_heads: int = 2, head_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads   = n_heads
        self.head_dim  = head_dim

        # Per-head attention projections
        self.q_projs = nn.ModuleList([
            nn.Linear(embed_dim, head_dim, bias=False) for _ in range(n_heads)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(embed_dim, head_dim, bias=False) for _ in range(n_heads)
        ])

        # Project multi-head context back to embed_dim
        self.out_proj = nn.Linear(n_heads * embed_dim, embed_dim, bias=False)
        # Gating: controls how much neighborhood context to incorporate
        self.gate_proj = nn.Linear(embed_dim * 2, embed_dim, bias=False)

    def forward(
        self,
        center_emb: torch.Tensor,         # [B, D]
        neighbor_emb: torch.Tensor,        # [B, K, D]
        neighbor_weights: torch.Tensor,    # [B, K]
        neighbor_mask: torch.Tensor,       # [B, K] bool: True = valid
    ) -> torch.Tensor:
        B, K, D = neighbor_emb.shape
        head_contexts = []

        for h in range(self.n_heads):
            # Compute query/key projections for this head
            q = self.q_projs[h](center_emb)       # [B, head_dim]
            k = self.k_projs[h](neighbor_emb)      # [B, K, head_dim]

            # Attention scores: q @ k.T / sqrt(head_dim) + log_confidence
            attn_scores = torch.bmm(
                q.unsqueeze(1), k.transpose(1, 2)
            ).squeeze(1) / (self.head_dim ** 0.5)  # [B, K]
            attn_scores = attn_scores + neighbor_weights
            attn_scores = attn_scores.masked_fill(~neighbor_mask, -1e9)
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, K]

            # Weighted aggregation of neighbor embeddings
            ctx_h = torch.bmm(attn_weights.unsqueeze(1), neighbor_emb).squeeze(1)  # [B, D]
            head_contexts.append(ctx_h)

        # Concatenate and project multi-head context
        multi_ctx = torch.cat(head_contexts, dim=-1)         # [B, n_heads*D]
        projected = self.out_proj(multi_ctx)                  # [B, D]

        # Gated fusion with center embedding
        gate_input = torch.cat([center_emb, projected], dim=-1)  # [B, 2D]
        gate = torch.sigmoid(self.gate_proj(gate_input))          # [B, D]
        output = gate * center_emb + (1 - gate) * projected       # [B, D]

        return output


# ---------------------------------------------------------------------------
# Focal Loss (key innovation — replaces weighted CE)
# ---------------------------------------------------------------------------
def focal_loss(
    logits: torch.Tensor,      # [N, C]
    targets: torch.Tensor,     # [N]
    class_weights: torch.Tensor,  # [C]
    gamma: float = 1.5,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """Focal loss with class weighting and optional label smoothing.

    Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    The (1 - p_t)^gamma factor downweights easy examples (well-classified
    neutral genes, p_t ≈ 0.95) and focuses training on hard minority examples
    (DEG up/down genes that the model is uncertain about).

    With gamma=1.5:
    - Well-classified neutral genes (p_t=0.95): weight = (0.05)^1.5 ≈ 0.011
    - Moderately correct DEG genes (p_t=0.7): weight = (0.30)^1.5 ≈ 0.164
    - Incorrectly classified DEG genes (p_t=0.3): weight = (0.70)^1.5 ≈ 0.586
    This provides 53× more focus on hard DEG examples vs. easy neutral genes.
    """
    C = logits.shape[1]

    # Apply label smoothing
    if label_smoothing > 0:
        n_classes = C
        smooth_val = label_smoothing / n_classes
        with torch.no_grad():
            soft_targets = torch.zeros_like(logits)
            soft_targets.fill_(smooth_val)
            soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing + smooth_val)
    else:
        soft_targets = None

    # Compute probabilities and log-probs
    log_probs = F.log_softmax(logits, dim=1)          # [N, C]
    probs = log_probs.exp()                            # [N, C]

    # Per-sample probability of true class (for focal weighting)
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]

    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1.0 - pt).pow(gamma)  # [N]

    # Cross-entropy loss (with or without label smoothing)
    if soft_targets is not None:
        ce_loss = -(soft_targets * log_probs).sum(dim=1)  # [N]
    else:
        ce_loss = F.nll_loss(log_probs, targets, reduction="none")  # [N]

    # Apply class weights (alpha factor)
    alpha = class_weights[targets]  # [N]

    # Final focal loss
    loss = alpha * focal_weight * ce_loss  # [N]
    return loss.mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOCellStringFusionModel(pl.LightningModule):
    """AIDO.Cell-100M LoRA (r=16) + STRING_GNN K=16 2-Head Attention + Focal Loss.

    Key improvements from parent lineage:
    1. AIDO.Cell-100M LoRA backbone (r=16, α=32): rich perturbation-aware
       transcriptional context from one-hot gene encoding
    2. STRING_GNN K=16 2-head neighborhood attention: PPI structural context
    3. Focal loss (γ=1.5): dynamically focuses on hard DEG minority examples
       without the catastrophic activation shock of GenePriorBias
    4. NO GenePriorBias: proven harmful across all tested architectures
    5. global_batch_size=256: fixes generalization gap observed in sibling

    Architecture:
        AIDO.Cell summary token [B, 640]
        STRING_GNN neighborhood [B, 256]
        → concat [B, 896]
        → head: Linear(896→256) → LN → GELU → Dropout(0.5) → Linear(256→19920)
        → view [B, 3, 6640]
    """

    def __init__(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        K: int = 16,
        n_heads: int = 2,
        head_dim: int = 128,
        head_hidden: int = 256,
        dropout: float = 0.5,
        lr: float = 1e-4,
        weight_decay: float = 2e-2,
        warmup_epochs: int = 10,
        t_max: int = 200,
        eta_min: float = 1e-6,
        label_smoothing: float = 0.05,
        focal_gamma: float = 1.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True

        hp = self.hparams
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # ----------------------------------------------------------------
        # 1. AIDO.Cell-100M tokenizer (rank 0 first, barrier, all load)
        # ----------------------------------------------------------------
        if local_rank == 0:
            AutoTokenizer.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)

        # ----------------------------------------------------------------
        # 2. AIDO.Cell-100M backbone with LoRA (r=16, trainable Q/K/V)
        # ----------------------------------------------------------------
        aido_backbone = AutoModel.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
        aido_backbone = aido_backbone.to(torch.bfloat16)

        # Patch get_input_embeddings to avoid NotImplementedError from PEFT's
        # enable_input_require_grads() call inside get_peft_model().
        # AIDO.Cell uses GeneEmbedding (custom, not word_embeddings) so we point
        # the stub to the actual gene_embedding layer.
        aido_backbone.get_input_embeddings = lambda: aido_backbone.bert.gene_embedding

        # Enable gradient checkpointing before LoRA wrapping
        aido_backbone.config.use_cache = False
        aido_backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            target_modules=["query", "key", "value"],
        )
        self.aido_cell = get_peft_model(aido_backbone, lora_cfg)

        # Cast LoRA params to float32 for stable optimization
        for name, param in self.aido_cell.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ----------------------------------------------------------------
        # 3. Pre-compute STRING_GNN node embeddings (backbone stays frozen)
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

        self.register_buffer("node_embeddings", node_emb)
        n_nodes = node_emb.shape[0]

        # ----------------------------------------------------------------
        # 4. Pre-compute top-K neighbors
        # ----------------------------------------------------------------
        self.print(f"Pre-computing top-{hp.K} PPI neighbors for {n_nodes} nodes...")
        nbr_idx, nbr_wgt = precompute_neighborhood(
            edge_index, edge_weight, n_nodes, K=hp.K
        )
        self.register_buffer("neighbor_indices", nbr_idx)  # [n_nodes, K]
        self.register_buffer("neighbor_weights", nbr_wgt)  # [n_nodes, K]

        del string_backbone, graph, edge_index, edge_weight, gnn_out

        # ----------------------------------------------------------------
        # 5. Learnable fallback for unknown pert_ids
        # ----------------------------------------------------------------
        self.fallback_emb = nn.Embedding(1, STRING_DIM)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)

        # ----------------------------------------------------------------
        # 6. STRING_GNN 2-head neighborhood attention (proven K=16, 2-head)
        # ----------------------------------------------------------------
        self.neighborhood_attn = MultiHeadNeighborhoodAttention(
            embed_dim=STRING_DIM,
            n_heads=hp.n_heads,
            head_dim=hp.head_dim,
        )

        # ----------------------------------------------------------------
        # 7. Fusion head: Linear(896→256) → LN → GELU → Dropout → Linear(256→19920)
        # ----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(FUSION_DIM, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )

        # Class weights for focal loss (sqrt-inverse-frequency)
        self.register_buffer("class_weights", get_class_weights())

        # Cast all remaining trainable parameters to float32
        for name, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Accumulators
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_meta:  List[Dict]         = []

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Trainable parameters: {n_trainable:,}")

    # ------------------------------------------------------------------
    # AIDO.Cell forward: tokenize and extract summary token
    # ------------------------------------------------------------------
    def _get_aido_embeddings(self, pert_ids: List[str]) -> torch.Tensor:
        """Tokenize perturbation identifiers and extract AIDO.Cell summary token.

        Each perturbation is encoded as a single-gene expression profile:
        - Perturbed gene: expression = 1.0 (indicating knockdown/knockout)
        - All other genes: -1.0 (missing/not measured)

        The summary token at position 19264 in last_hidden_state captures the
        model's learned representation of this perturbation context.

        Args:
            pert_ids: list of Ensembl gene IDs, e.g. ['ENSG00000012048', ...]
        Returns:
            [B, AIDO_DIM] float32 summary embeddings
        """
        # Tokenize: each sample has one gene expressed at 1.0
        input_dicts = [
            {"gene_ids": [pid], "expression": [1.0]}
            for pid in pert_ids
        ]
        tokenized = self.tokenizer(input_dicts, return_tensors="pt")
        # input_ids: [B, 19264] float32 — value 1.0 for perturbed gene, -1.0 for others
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        # Ensure float32 for LoRA parameters during forward
        if input_ids.dtype != torch.float32:
            input_ids = input_ids.float()

        outputs = self.aido_cell(input_ids=input_ids, attention_mask=attention_mask)

        # Summary token: position 19264 (the second-to-last appended token)
        # last_hidden_state shape: [B, 19266, 640]
        # Position 19264 is the first summary token appended by _prepare_inputs
        summary = outputs.last_hidden_state[:, 19264, :].float()  # [B, 640]
        return summary

    # ------------------------------------------------------------------
    # STRING_GNN embedding lookup with 2-head neighborhood aggregation
    # ------------------------------------------------------------------
    def _get_string_embeddings(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Lookup pre-computed embeddings with 2-head PPI neighborhood aggregation.

        Args:
            string_node_idx: [B] long tensor, -1 for pert_ids not in STRING.
        Returns:
            [B, STRING_DIM] aggregated perturbation embeddings.
        """
        B = string_node_idx.shape[0]
        dev = self.node_embeddings.device
        K   = self.neighbor_indices.shape[1]

        # Base center embeddings
        center_emb = torch.zeros(B, STRING_DIM, dtype=torch.float32, device=dev)
        known   = string_node_idx >= 0
        unknown = ~known

        if known.any():
            center_emb[known] = self.node_embeddings[string_node_idx[known]]
        if unknown.any():
            fb = self.fallback_emb(
                torch.zeros(unknown.sum(), dtype=torch.long, device=dev)
            ).float()
            center_emb[unknown] = fb

        output_emb = center_emb.clone()

        if known.any():
            known_idx = string_node_idx[known]

            nbr_idx = self.neighbor_indices[known_idx]   # [B_known, K]
            nbr_wgt = self.neighbor_weights[known_idx]   # [B_known, K]
            nbr_msk = nbr_idx >= 0                       # [B_known, K] valid mask

            nbr_idx_safe = nbr_idx.clamp(min=0)
            nbr_emb = self.node_embeddings[nbr_idx_safe]  # [B_known, K, D]
            nbr_emb = nbr_emb * nbr_msk.unsqueeze(-1).float()

            aggregated = self.neighborhood_attn(
                center_emb[known], nbr_emb, nbr_wgt, nbr_msk
            )  # [B_known, D]
            output_emb[known] = aggregated

        return output_emb

    def forward(
        self,
        pert_ids: List[str],
        string_node_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        # AIDO.Cell summary token [B, 640]
        aido_emb = self._get_aido_embeddings(pert_ids)

        # STRING_GNN neighborhood embedding [B, 256]
        string_emb = self._get_string_embeddings(string_node_idx)

        # Fusion: concat [B, 896]
        fusion = torch.cat([aido_emb, string_emb], dim=-1)

        # Head: [B, 3*G] → view [B, 3, G]
        logits = self.head(fusion).view(-1, N_CLASSES, N_GENES)

        return logits

    # ------------------------------------------------------------------
    # Focal Loss
    # ------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        logits_flat  = logits.permute(0, 2, 1).reshape(-1, C)   # [B*G, 3]
        targets_flat = targets.reshape(-1)                        # [B*G]

        loss = focal_loss(
            logits_flat,
            targets_flat,
            class_weights=self.class_weights,
            gamma=self.hparams.focal_gamma,
            label_smoothing=self.hparams.label_smoothing,
        )
        return loss

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["pert_id"], batch["string_node_idx"])
        loss = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["pert_id"], batch["string_node_idx"])
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
        local_preds = torch.cat(self._val_preds, dim=0)   # [N_local, 3, G]
        local_tgts  = torch.cat(self._val_tgts,  dim=0)   # [N_local, G]
        local_idx   = torch.cat(self._val_idx,   dim=0)   # [N_local]
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds)   # [W, N_local, 3, G]
        all_tgts  = self.all_gather(local_tgts)    # [W, N_local, G]
        all_idx   = self.all_gather(local_idx)     # [W, N_local]

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

        # De-duplicate
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
        logits = self(batch["pert_id"], batch["string_node_idx"])
        probs  = torch.softmax(logits, dim=1).detach()
        self._test_preds.append(probs)
        meta = [
            {"sample_idx": int(i.item()), "pert_id": p, "symbol": s}
            for i, p, s in zip(
                batch["sample_idx"], batch["pert_id"], batch["symbol"]
            )
        ]
        self._test_meta.extend(meta)
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds  = torch.cat(self._test_preds, dim=0)  # [N_local, 3, G]
        local_idx_t  = torch.tensor(
            [m["sample_idx"] for m in self._test_meta], dtype=torch.long,
            device=local_preds.device,
        )

        all_preds = self.all_gather(local_preds)   # [W, N_local, 3, G]
        all_idx   = self.all_gather(local_idx_t)   # [W, N_local]

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            pred_map: Dict[int, torch.Tensor] = {}
            for i in range(len(idx_flat)):
                gid = int(idx_flat[i].item())
                if gid not in pred_map:
                    pred_map[gid] = preds_flat[i]

            test_df = pd.read_csv(TEST_TSV, sep="\t")
            rows = []
            for i in range(len(test_df)):
                if i not in pred_map:
                    continue
                pid  = test_df.iloc[i]["pert_id"]
                sym  = test_df.iloc[i]["symbol"]
                pred = pred_map[i].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            self.print(f"[Node1-2] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_meta.clear()

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
        buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {train}/{total} params "
            f"({100 * train / total:.2f}%), plus {buffers} buffer values"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        trainable_keys = {n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys = {n for n, _ in self.named_buffers()}
        expected = trainable_keys | buffer_keys
        missing = [k for k in expected if k not in state_dict]
        unexpected = [k for k in state_dict if k not in expected]
        if missing:
            self.print(f"Warning: Missing checkpoint keys: {missing[:5]}...")
        if unexpected:
            self.print(f"Warning: Unexpected keys: {unexpected[:5]}...")
        loaded_t = len([k for k in state_dict if k in trainable_keys])
        loaded_b = len([k for k in state_dict if k in buffer_keys])
        self.print(f"Loading checkpoint: {loaded_t} trainable params + {loaded_b} buffers")
        return super().load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Optimizer: AdamW with separate LRs + warmup + CosineAnnealingLR
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams

        # Separate parameter groups for LoRA (lower LR) vs. head/string (full LR)
        aido_lora_params = []
        other_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "aido_cell" in name:
                aido_lora_params.append(p)
            else:
                other_params.append(p)

        param_groups = [
            {"params": aido_lora_params, "lr": hp.lr * 0.5},    # LoRA gets half LR
            {"params": other_params, "lr": hp.lr},               # Head + string attention: full LR
        ]

        opt = torch.optim.AdamW(param_groups, weight_decay=hp.weight_decay)

        # Phase 1: linear warmup from 0.1×lr → lr over warmup_epochs
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_epochs,
        )
        # Phase 2: cosine annealing from lr → eta_min over t_max epochs
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=hp.t_max,
            eta_min=hp.eta_min,
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
        description="Node1-2 – AIDO.Cell-100M LoRA (r=16) + STRING_GNN K=16 2-Head + Focal Loss"
    )
    parser.add_argument("--micro-batch-size",    type=int,   default=4)
    parser.add_argument("--global-batch-size",   type=int,   default=256)
    parser.add_argument("--max-epochs",          type=int,   default=300)
    parser.add_argument("--lr",                  type=float, default=1e-4)
    parser.add_argument("--weight-decay",        type=float, default=2e-2)
    parser.add_argument("--lora-r",              type=int,   default=16,
                        dest="lora_r")
    parser.add_argument("--lora-alpha",          type=int,   default=32,
                        dest="lora_alpha")
    parser.add_argument("--lora-dropout",        type=float, default=0.05,
                        dest="lora_dropout")
    parser.add_argument("--k",                   type=int,   default=16)
    parser.add_argument("--n-heads",             type=int,   default=2,
                        dest="n_heads")
    parser.add_argument("--head-dim",            type=int,   default=128,
                        dest="head_dim")
    parser.add_argument("--head-hidden",         type=int,   default=256,
                        dest="head_hidden")
    parser.add_argument("--dropout",             type=float, default=0.5)
    parser.add_argument("--warmup-epochs",       type=int,   default=10)
    parser.add_argument("--t-max",               type=int,   default=200,
                        dest="t_max")
    parser.add_argument("--eta-min",             type=float, default=1e-6,
                        dest="eta_min")
    parser.add_argument("--label-smoothing",     type=float, default=0.05,
                        dest="label_smoothing")
    parser.add_argument("--focal-gamma",         type=float, default=1.5,
                        dest="focal_gamma")
    parser.add_argument("--num-workers",         type=int,   default=4)
    parser.add_argument("--val-check-interval",  type=float, default=1.0,
                        dest="val_check_interval")
    parser.add_argument("--gradient-clip-val",   type=float, default=1.0,
                        dest="gradient_clip_val")
    parser.add_argument("--debug-max-step",      type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",        action="store_true",
                        dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = args.debug_max_step
        lim_val   = args.debug_max_step
        lim_test  = 1.0       # ALWAYS run full test set; --debug-max-step limits train/val only
        max_steps = args.debug_max_step
    else:
        lim_train = 1.0
        lim_val   = 1.0
        lim_test  = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    model = AIDOCellStringFusionModel(
        lora_r          = args.lora_r,
        lora_alpha      = args.lora_alpha,
        lora_dropout    = args.lora_dropout,
        K               = args.k,
        n_heads         = args.n_heads,
        head_dim        = args.head_dim,
        head_hidden     = args.head_hidden,
        dropout         = args.dropout,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        warmup_epochs   = args.warmup_epochs,
        t_max           = args.t_max,
        eta_min         = args.eta_min,
        label_smoothing = args.label_smoothing,
        focal_gamma     = args.focal_gamma,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
    )
    # patience=20: ensures we capture late improvement spikes in AIDO.Cell training
    # (node2-1-1-1-1-1 feedback: "patience=10 was barely sufficient for late spike at epoch 77")
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=20, min_delta=1e-4)
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
        val_check_interval      = args.val_check_interval if (args.debug_max_step is None and not fast_dev_run) else 1.0,
        num_sanity_val_steps    = 2,
        callbacks               = [ckpt_cb, es_cb, lr_cb, pg_cb],
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = args.gradient_clip_val,
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
