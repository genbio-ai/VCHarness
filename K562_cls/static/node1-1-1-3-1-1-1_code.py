"""Node 1-1-1-3-1-1-1: AIDO.Cell-100M LoRA + STRING_GNN K=16 2-Head + Multi-Checkpoint Ensemble.

Recovery and improvement from node1-1-1-3-1-1 (F1=0.3669) catastrophic failure.
Based on proven node2-1-1-1-1-1 architecture (F1=0.5128).

ROOT CAUSE OF PARENT FAILURE:
  1. GenePriorBias activation at epoch 50 caused -16% relative val F1 drop (0.4414→0.3718)
  2. global_batch_size=128 (vs proven 256) caused generalization gap (val=0.4414, test=0.3669)

KEY CHANGES FROM PARENT (node1-1-1-3-1-1, F1=0.3669):
  1. REMOVE GenePriorBias — proven catastrophically harmful in every tested context
  2. global_batch_size=256 — fixes generalization gap (matches node2-1-1-1-1-1's zero gap)
  3. Multi-checkpoint ensemble at test time (top-3 by val/f1, average softmax outputs)
     → Proven +0.010 F1 in node4-2-1-2-1 (scFoundation context)
  4. patience=20 — captures late improvement spikes
     → node2-1-1-1-1-1 peaked at epoch 77 with patience=10 that "barely sufficient"
  5. max_epochs=300 — full convergence budget
     → node2-1-1-1-1-1 feedback: "grandparent not converged at epoch 63"

ARCHITECTURE (identical to node2-1-1-1-1-1, best in tree at F1=0.5128):
  Stream 1: AIDO.Cell-100M (LoRA r=8, α=16)
    input_ids [B, 19264] → 18-layer BERT-like transformer → lhs[B, 19266, 640]
    summary = lhs[:, 19264, :]  → [B, 640]

  Stream 2: STRING_GNN K=16 2-head neighborhood attention (frozen)
    pert_id → center_emb [B, 256] + top-16 PPI neighbors → aggregated [B, 256]

  Fusion: concat([summary, string_ctx]) → [B, 896]

  Head: Linear(896→256) → LayerNorm → GELU → Dropout(0.5) → Linear(256→19920)
        → view [B, 3, 6640]

  NO GenePriorBias.

EXPECTED: F1 > 0.515 (recovery to 0.5128 + ensemble bonus ~+0.005-0.010)
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

# Remapped class frequencies (after -1→0, 0→1, 1→2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

AIDO_CELL_DIR = Path("/home/Models/AIDO.Cell-100M")
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

AIDO_DIM   = 640   # AIDO.Cell-100M hidden_size
STRING_DIM = 256   # STRING_GNN hidden dim
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
        if c > 0:
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


def save_predictions(
    pred_map: Dict[int, torch.Tensor],
    test_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Save test predictions dict to TSV file."""
    rows = []
    for i in range(len(test_df)):
        if i not in pred_map:
            continue
        pid  = test_df.iloc[i]["pert_id"]
        sym  = test_df.iloc[i]["symbol"]
        pred = pred_map[i].float().cpu().numpy().tolist()
        rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred)})
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"[Node1-1-1-3-1-1-1] Saved {len(rows)} test predictions to {out_path}.")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    """K562 DEG prediction dataset with pre-tokenized AIDO.Cell inputs."""

    def __init__(
        self,
        df: pd.DataFrame,
        string_map: Dict[str, int],
        tokenizer,
    ) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()

        # STRING_GNN node index for each sample (-1 means not in STRING)
        self.string_node_indices = torch.tensor(
            [string_map.get(p, -1) for p in self.pert_ids], dtype=torch.long
        )

        # Pre-tokenize AIDO.Cell inputs: one-hot perturbation encoding
        # Perturbed gene = 1.0, all others = -1.0 (missing)
        input_ids_list = []
        for pert_id in self.pert_ids:
            inp = tokenizer(
                {"gene_ids": [pert_id], "expression": [1.0]},
                return_tensors="pt",
            )
            input_ids_list.append(inp["input_ids"].squeeze(0))  # [19264] float32
        self.aido_input_ids = torch.stack(input_ids_list)  # [N, 19264] float32

        # Labels
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
            "aido_input_ids":  self.aido_input_ids[idx],  # [19264] float32
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]  # [G] in {0, 1, 2}
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_idx":      torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
        "pert_id":         [b["pert_id"] for b in batch],
        "symbol":          [b["symbol"]  for b in batch],
        "string_node_idx": torch.stack([b["string_node_idx"] for b in batch]),
        "aido_input_ids":  torch.stack([b["aido_input_ids"]  for b in batch]),  # [B, 19264]
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
        self.tokenizer   = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.string_map is not None:
            return  # already set up

        self.string_map = load_string_gnn_mapping()

        # Rank-0 downloads first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df, self.string_map, self.tokenizer)
        self.val_ds   = DEGDataset(val_df,   self.string_map, self.tokenizer)
        self.test_ds  = DEGDataset(test_df,  self.string_map, self.tokenizer)

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
# Multi-Head Neighborhood Attention
# ---------------------------------------------------------------------------
class MultiHeadNeighborhoodAttentionAggregator(nn.Module):
    """K=16 PPI neighborhood attention with 2 heads.

    Proven architecture from node2-1-1-1-1-1 (F1=0.5128):
      Head 1 & 2: q=W_q(center), k=W_k(neigh), attn=softmax(q@k.T/sqrt(head_dim) + log_conf)
      context_h = attn @ neigh  # [B, D] per head
      multi_context = concat([ctx1, ctx2])    # [B, 2D]
      projected = W_out(multi_context)         # [B, D]
      gate = sigmoid(W_gate(concat(center, projected)))  # [B, D]
      output = gate * center + (1-gate) * projected
    """

    def __init__(self, embed_dim: int = 256, n_heads: int = 2) -> None:
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads   = n_heads
        self.head_dim  = embed_dim // n_heads  # 128 with dim=256, heads=2

        # Query/key projections per head
        self.W_q = nn.ModuleList([nn.Linear(embed_dim, self.head_dim, bias=False) for _ in range(n_heads)])
        self.W_k = nn.ModuleList([nn.Linear(embed_dim, self.head_dim, bias=False) for _ in range(n_heads)])

        # Output projection: concat(n_heads * head_dim) → embed_dim
        self.W_out  = nn.Linear(n_heads * embed_dim, embed_dim, bias=False)
        # Gating: [center, projected] → gate
        self.W_gate = nn.Linear(embed_dim * 2, embed_dim, bias=True)
        self.scale  = self.head_dim ** -0.5

    def forward(
        self,
        center_emb:       torch.Tensor,  # [B, D]
        neighbor_emb:     torch.Tensor,  # [B, K, D]
        neighbor_weights: torch.Tensor,  # [B, K]
        neighbor_mask:    torch.Tensor,  # [B, K] bool: True = valid
    ) -> torch.Tensor:
        B, K, D = neighbor_emb.shape
        log_conf = torch.log(neighbor_weights.clamp(min=1e-8))  # [B, K]
        log_conf = log_conf.masked_fill(~neighbor_mask, -1e9)

        contexts = []
        for h in range(self.n_heads):
            q = self.W_q[h](center_emb)              # [B, head_dim]
            k = self.W_k[h](neighbor_emb)            # [B, K, head_dim]
            # Attention score: q dot k_i / sqrt(head_dim) + log_conf
            attn = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) * self.scale  # [B, K]
            attn = attn + log_conf
            attn = attn.masked_fill(~neighbor_mask, -1e9)
            attn = torch.softmax(attn, dim=-1)       # [B, K]
            ctx  = torch.bmm(attn.unsqueeze(1), neighbor_emb).squeeze(1)  # [B, D]
            contexts.append(ctx)

        multi_ctx  = torch.cat(contexts, dim=-1)                     # [B, n_heads*D]
        projected  = self.W_out(multi_ctx)                           # [B, D]
        gate       = torch.sigmoid(
            self.W_gate(torch.cat([center_emb, projected], dim=-1))  # [B, D]
        )
        return gate * center_emb + (1.0 - gate) * projected


# ---------------------------------------------------------------------------
# Lightning Model — NO GenePriorBias
# ---------------------------------------------------------------------------
class AIDOCellStringFusionModel(pl.LightningModule):
    """AIDO.Cell-100M LoRA + STRING_GNN K=16 2-Head Neighborhood Attention.

    Architecture (identical to node2-1-1-1-1-1, best in tree at F1=0.5128):
      Stream 1: AIDO.Cell-100M (LoRA r=8, α=16)
        input_ids [B, 19264] float32 → transformer → lhs[B, 19266, 640]
        summary = lhs[:, 19264, :]  → [B, 640]

      Stream 2: STRING_GNN K=16 2-head neighborhood attention (frozen)
        pert_id → center_emb [B, 256] + top-K neighbors → [B, 256]

      Fusion: concat([summary, string_ctx]) → [B, 896]

      Head: Linear(896→256) → LN → GELU → Dropout(0.5) → Linear(256→19920)
            → view [B, 3, 6640]

    NO GenePriorBias (removed — proven catastrophically harmful).
    """

    def __init__(
        self,
        head_hidden:       int   = 256,
        head_dropout:      float = 0.5,
        K:                 int   = 16,
        lora_r:            int   = 8,
        lora_alpha:        int   = 16,
        lr:                float = 1e-4,
        weight_decay:      float = 2e-2,
        warmup_epochs:     int   = 10,
        t_max:             int   = 200,
        eta_min:           float = 1e-6,
        label_smoothing:   float = 0.05,
        gradient_clip_val: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True
        hp = self.hparams

        # ----------------------------------------------------------------
        # 1. AIDO.Cell-100M backbone with LoRA
        # ----------------------------------------------------------------
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Load config and enable FlashAttention-2 (critical for seq_len=19266).
        # Without this, standard attention allocates O(n²) memory (~21 GB/layer) and OOMs.
        from transformers import AutoConfig
        aido_config = AutoConfig.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
        aido_config._use_flash_attention_2 = True
        aido_base = AutoModel.from_pretrained(str(AIDO_CELL_DIR), config=aido_config, trust_remote_code=True)
        aido_base.config.use_cache = False
        aido_base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # --- Patch: AIDO.Cell.get_input_embeddings() raises NotImplementedError,
        # but PEFT's get_peft_model() calls enable_input_require_grads() which needs it.
        # Patch the method so PEFT can register its forward-hook on the gene embedding.
        def _get_input_emb(self):
            return getattr(self, "gene_embedding", self)
        aido_base.__class__.get_input_embeddings = _get_input_emb

        # CRITICAL: Convert ALL fp32 parameters to bf16 so the model operates fully in bf16.
        # Without this, the attention LayerNorm outputs fp32 (LayerNorm uses fp32 accumulation),
        # which causes CellFoundationAttention.forward() to skip FlashAttention and use
        # standard O(n²) attention → OOM at seq_len=19266.
        # All base-model params (including LayerNorm, embeddings, QKV) are converted to bf16.
        # LoRA params are added on top in float32 (see below) for stable optimization.
        for name, param in aido_base.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=0.05,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,
        )
        self.aido_cell = get_peft_model(aido_base, lora_cfg)

        # Cast LoRA params to float32 for stable optimization
        for name, param in self.aido_cell.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ----------------------------------------------------------------
        # 2. Pre-compute STRING_GNN node embeddings (frozen)
        # ----------------------------------------------------------------
        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index  = graph["edge_index"].long()
        edge_weight = graph["edge_weight"].float()

        with torch.no_grad():
            gnn_out  = backbone(edge_index=edge_index, edge_weight=edge_weight)
            node_emb = gnn_out.last_hidden_state.float().detach()  # [18870, 256]

        self.register_buffer("node_embeddings", node_emb)
        n_nodes = node_emb.shape[0]

        # Pre-compute top-K neighbors
        self.print(f"Pre-computing top-{hp.K} PPI neighbors for {n_nodes} nodes...")
        nbr_idx, nbr_wgt = precompute_neighborhood(
            edge_index, edge_weight, n_nodes, K=hp.K
        )
        self.register_buffer("neighbor_indices", nbr_idx)  # [n_nodes, K]
        self.register_buffer("neighbor_weights", nbr_wgt)  # [n_nodes, K]
        del backbone, graph, edge_index, edge_weight, gnn_out

        # Fallback embedding for unknown pert_ids
        self.fallback_emb = nn.Embedding(1, STRING_DIM)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)
        self.fallback_emb.weight.data = self.fallback_emb.weight.data.float()

        # ----------------------------------------------------------------
        # 3. Multi-head (2-head) neighborhood attention
        # ----------------------------------------------------------------
        self.neighborhood_attn = MultiHeadNeighborhoodAttentionAggregator(
            embed_dim=STRING_DIM,
            n_heads=2,
        )

        # ----------------------------------------------------------------
        # 4. Fusion head: Linear(896→256) → LN → GELU → Dropout → Linear(256→19920)
        # ----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(FUSION_DIM, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )

        # Class weights for weighted CE (sqrt-inverse-frequency)
        self.register_buffer("class_weights", get_class_weights())

        # Cast all remaining trainable params to float32
        for name, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Accumulators
        self._val_preds:   List[torch.Tensor] = []
        self._val_tgts:    List[torch.Tensor] = []
        self._val_idx:     List[torch.Tensor] = []
        self._test_preds:  List[torch.Tensor] = []
        self._test_meta:   List[Dict]         = []
        # Populated in on_test_epoch_end (rank 0 only), used for ensemble
        self._test_pred_map: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # STRING_GNN neighborhood aggregation
    # ------------------------------------------------------------------
    def _get_string_embeddings(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Lookup frozen STRING_GNN embeddings with 2-head PPI attention.

        Args:
            string_node_idx: [B] long, -1 for unknowns.
        Returns:
            [B, STRING_DIM] contextual embeddings.
        """
        B   = string_node_idx.shape[0]
        dev = self.node_embeddings.device
        K   = self.neighbor_indices.shape[1]

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
            known_idx    = string_node_idx[known]
            nbr_idx      = self.neighbor_indices[known_idx]    # [B_k, K]
            nbr_wgt      = self.neighbor_weights[known_idx]    # [B_k, K]
            nbr_msk      = nbr_idx >= 0                        # [B_k, K] valid mask
            nbr_idx_safe = nbr_idx.clamp(min=0)
            nbr_emb      = self.node_embeddings[nbr_idx_safe]  # [B_k, K, D]
            # Zero out padding
            nbr_emb = nbr_emb * nbr_msk.unsqueeze(-1).float()

            aggregated = self.neighborhood_attn(
                center_emb[known], nbr_emb, nbr_wgt, nbr_msk
            )  # [B_k, D]
            output_emb[known] = aggregated

        return output_emb  # [B, STRING_DIM]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        aido_input_ids:  torch.Tensor,  # [B, 19264] float32
        string_node_idx: torch.Tensor,  # [B] long
    ) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        B = aido_input_ids.shape[0]

        # --- AIDO.Cell stream ---
        aido_inputs = {
            "input_ids":      aido_input_ids,
            "attention_mask": torch.ones(B, 19264, dtype=torch.long,
                                         device=aido_input_ids.device),
        }
        aido_out    = self.aido_cell(**aido_inputs)
        # Summary token at position 19264 (first of 2 appended summary tokens)
        summary_emb = aido_out.last_hidden_state[:, 19264, :].float()  # [B, 640]

        # --- STRING_GNN stream ---
        string_emb = self._get_string_embeddings(string_node_idx)      # [B, 256]

        # --- Fusion ---
        fused  = torch.cat([summary_emb, string_emb], dim=-1)          # [B, 896]
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)          # [B, 3, G]

        # No GenePriorBias (removed — catastrophically harmful in all tested contexts)
        return logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        logits_flat  = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
        targets_flat = targets.reshape(-1)                       # [B*G]
        return F.cross_entropy(
            logits_flat,
            targets_flat,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["aido_input_ids"], batch["string_node_idx"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True,
                 on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["aido_input_ids"], batch["string_node_idx"])
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

        # De-duplicate across DDP ranks
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

    def on_test_epoch_start(self) -> None:
        """Reset test prediction storage before each test run (supports multi-run ensemble)."""
        self._test_preds = []
        self._test_meta  = []
        self._test_pred_map = {}

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["aido_input_ids"], batch["string_node_idx"])
        probs  = torch.softmax(logits, dim=1).detach()
        self._test_preds.append(probs)
        meta = [
            {"sample_idx": int(i.item()), "pert_id": p, "symbol": s}
            for i, p, s in zip(batch["sample_idx"], batch["pert_id"], batch["symbol"])
        ]
        self._test_meta.extend(meta)
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds = torch.cat(self._test_preds, dim=0)  # [N_local, 3, G]
        local_idx_t = torch.tensor(
            [m["sample_idx"] for m in self._test_meta],
            dtype=torch.long, device=local_preds.device,
        )

        all_preds = self.all_gather(local_preds)
        all_idx   = self.all_gather(local_idx_t)

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            # De-duplicate and store in _test_pred_map (for ensemble logic in main)
            pred_map: Dict[int, torch.Tensor] = {}
            for i in range(len(idx_flat)):
                gid = int(idx_flat[i].item())
                if gid not in pred_map:
                    pred_map[gid] = preds_flat[i].float().cpu()

            self._test_pred_map = pred_map
            self.print(f"[TestEpochEnd] Collected {len(pred_map)} predictions.")

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
        bufs  = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Checkpoint: {train}/{total} params ({100*train/total:.1f}%), "
            f"{bufs} buffer values"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Optimizer: AdamW + linear warmup + CosineAnnealingLR
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=hp.lr, weight_decay=hp.weight_decay)

        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=hp.warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=hp.t_max, eta_min=hp.eta_min,
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
        description="Node1-1-1-3-1-1-1 – AIDO.Cell-100M LoRA + STRING_GNN K=16 2-Head + Ensemble"
    )
    parser.add_argument("--micro-batch-size",    type=int,   default=16)
    # global_batch_size=256 matches node2-1-1-1-1-1 (critical for generalization)
    parser.add_argument("--global-batch-size",   type=int,   default=256)
    parser.add_argument("--max-epochs",          type=int,   default=300)
    parser.add_argument("--lr",                  type=float, default=1e-4)
    parser.add_argument("--weight-decay",        type=float, default=2e-2)
    parser.add_argument("--head-hidden",         type=int,   default=256,
                        dest="head_hidden")
    parser.add_argument("--head-dropout",        type=float, default=0.5,
                        dest="head_dropout")
    parser.add_argument("--k",                   type=int,   default=16)
    parser.add_argument("--lora-r",              type=int,   default=8,
                        dest="lora_r")
    parser.add_argument("--lora-alpha",          type=int,   default=16,
                        dest="lora_alpha")
    parser.add_argument("--warmup-epochs",       type=int,   default=10,
                        dest="warmup_epochs")
    parser.add_argument("--t-max",               type=int,   default=200,
                        dest="t_max")
    parser.add_argument("--eta-min",             type=float, default=1e-6,
                        dest="eta_min")
    parser.add_argument("--label-smoothing",     type=float, default=0.05,
                        dest="label_smoothing")
    parser.add_argument("--gradient-clip-val",   type=float, default=1.0,
                        dest="gradient_clip_val")
    parser.add_argument("--num-workers",         type=int,   default=4)
    parser.add_argument("--val-check-interval",  type=float, default=1.0,
                        dest="val_check_interval")
    # patience=20: captures late improvement spikes (node2-1-1-1-1-1 peaked at epoch 77)
    parser.add_argument("--patience",            type=int,   default=20)
    # save_top_k=3 for checkpoint ensemble
    parser.add_argument("--save-top-k",          type=int,   default=3,
                        dest="save_top_k")
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
        lim_test  = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = 1.0
        lim_val   = 1.0
        lim_test  = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)

    model = AIDOCellStringFusionModel(
        head_hidden        = args.head_hidden,
        head_dropout       = args.head_dropout,
        K                  = args.k,
        lora_r             = args.lora_r,
        lora_alpha         = args.lora_alpha,
        lr                 = args.lr,
        weight_decay       = args.weight_decay,
        warmup_epochs      = args.warmup_epochs,
        t_max              = args.t_max,
        eta_min            = args.eta_min,
        label_smoothing    = args.label_smoothing,
        gradient_clip_val  = args.gradient_clip_val,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = args.save_top_k,  # Save top-3 for ensemble
        auto_insert_metric_name=False,
    )
    es_cb = EarlyStopping(
        monitor   = "val/f1",
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
        val_check_interval      = args.val_check_interval if (args.debug_max_step is None and not fast_dev_run) else 1.0,
        num_sanity_val_steps    = 2,
        callbacks               = [ckpt_cb, es_cb, lr_cb, pg_cb],
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        gradient_clip_val       = args.gradient_clip_val,
        fast_dev_run            = fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)

    # -----------------------------------------------------------------------
    # Test Phase: Multi-checkpoint ensemble OR single best checkpoint
    # -----------------------------------------------------------------------
    test_df = pd.read_csv(TEST_TSV, sep="\t")

    if fast_dev_run or args.debug_max_step is not None:
        # Debug mode: single test pass
        trainer.test(model, datamodule=dm)
        if trainer.is_global_zero and model._test_pred_map:
            save_predictions(model._test_pred_map, test_df, output_dir / "test_predictions.tsv")
    else:
        # Production mode: multi-checkpoint ensemble
        best_k_models = ckpt_cb.best_k_models  # {path: val_f1_score}

        if len(best_k_models) >= 2:
            # Sort checkpoints by val/f1 (descending) and take top-3
            sorted_ckpts = sorted(
                best_k_models.keys(),
                key=lambda k: best_k_models[k],
                reverse=True,
            )[:3]

            print(f"\n[Ensemble] Running test with {len(sorted_ckpts)} checkpoints:")
            for i, cp in enumerate(sorted_ckpts):
                print(f"  [{i+1}] {Path(cp).name}  val/f1={best_k_models[cp]:.4f}")

            ensemble_pred_maps = []
            for ckpt_path in sorted_ckpts:
                # Each trainer.test() call loads the checkpoint and runs forward pass.
                # on_test_epoch_start resets _test_pred_map before each run.
                trainer.test(model, datamodule=dm, ckpt_path=str(ckpt_path))
                if trainer.is_global_zero and model._test_pred_map:
                    # Clone to avoid mutation in subsequent test() calls
                    ensemble_pred_maps.append(
                        {k: v.clone() for k, v in model._test_pred_map.items()}
                    )

            if trainer.is_global_zero and ensemble_pred_maps:
                # Average softmax predictions across checkpoints
                all_indices = set()
                for pm in ensemble_pred_maps:
                    all_indices.update(pm.keys())

                avg_pred_map: Dict[int, torch.Tensor] = {}
                for idx in all_indices:
                    preds_list = [pm[idx] for pm in ensemble_pred_maps if idx in pm]
                    avg_pred_map[idx] = torch.stack(preds_list).mean(dim=0)

                save_predictions(avg_pred_map, test_df, output_dir / "test_predictions.tsv")
                print(f"[Ensemble] Averaged {len(ensemble_pred_maps)} checkpoints → "
                      f"{len(avg_pred_map)} predictions saved.")
            elif trainer.is_global_zero:
                # Fallback: no pred maps collected (shouldn't happen)
                print("[Ensemble] Warning: No pred maps collected; falling back to best ckpt.")
                trainer.test(model, datamodule=dm, ckpt_path="best")
                if model._test_pred_map:
                    save_predictions(model._test_pred_map, test_df, output_dir / "test_predictions.tsv")
        else:
            # Only 1 checkpoint available: use best
            print("[Test] Using best single checkpoint (< 2 checkpoints saved).")
            trainer.test(model, datamodule=dm, ckpt_path="best")
            if trainer.is_global_zero and model._test_pred_map:
                save_predictions(model._test_pred_map, test_df, output_dir / "test_predictions.tsv")


if __name__ == "__main__":
    main()
