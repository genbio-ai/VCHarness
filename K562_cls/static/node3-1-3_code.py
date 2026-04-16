"""Node 3-1-3 – STRING_GNN K=16 Neighborhood Attention + AIDO.Cell-10M + QKV + Muon.

Differentiation from siblings:
- node3-1-1 (F1=0.4325): AIDO.Cell-only, no STRING_GNN, QKV+Output, CosineWR T_0=50
- node3-1-2 (F1=0.4407): frozen STRING_GNN **direct lookup** (no neighborhood), SGDR T_0=15

This node implements:
1. K=16 neighborhood attention (attn_dim=64, center-context gating) — proven recipe:
   node1-1-1-1-1 achieved F1=0.4846 (vs 0.4407 for direct lookup in sibling)
2. AIDO.Cell-10M QKV-only fine-tuning (Muon) — transcriptome context
3. Standard CosineAnnealingLR (T_max=max_epochs) — no warm restarts, avoids
   the epoch-15 disruption that dropped sibling node3-1-2 from 0.441→0.433
4. Smaller head (head_hidden=256, dropout=0.4) — reduces overfitting on 1388 samples
5. Label-smoothed CE + class weights — proven stable (avoids focal loss collapse)
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
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES         = 6640
N_CLASSES       = 3
AIDO_GENES      = 19264
AIDO_MODEL_DIR  = "/home/Models/AIDO.Cell-10M"
STRING_GNN_DIR  = "/home/Models/STRING_GNN"
HIDDEN_DIM      = 256    # AIDO.Cell-10M hidden size
N_AIDO_LAYERS   = 8      # AIDO.Cell-10M transformer depth
K_NEIGHBORS     = 16     # top-K PPI neighbors (proven best, from node1-1-1-1-1 F1=0.4846)
ATTN_DIM        = 64     # neighborhood attention dimension (proven best over 32)

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
    """Sqrt-inverse-frequency class weights for handling 92.5% neutral imbalance."""
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
    y_hat       = preds.argmax(dim=1)   # [N, G]
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
        f1_c = torch.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-8),
                            torch.zeros_like(prec))
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# Neighborhood Attention with Center-Context Gating
# ---------------------------------------------------------------------------
class NeighborhoodAttentionGated(nn.Module):
    """Attention-weighted PPI neighborhood aggregation with center-context gating.

    Architecture:
    1. Compute attention weights over K PPI neighbors using gene as query
    2. Aggregate neighbors into a context vector
    3. Blend center (gene self) and context (aggregated neighbors) with a learned gate

    This is the proven architecture from node1-1-1-1-1 which achieved F1=0.4846.
    Key hyperparameters: K=16, attn_dim=64 (attn_dim=32 degraded to 0.4743).
    """
    def __init__(self, dim: int = 256, attn_dim: int = 64):
        super().__init__()
        self.q_proj    = nn.Linear(dim, attn_dim, bias=False)
        self.k_proj    = nn.Linear(dim, attn_dim, bias=False)
        self.scale     = attn_dim ** -0.5
        # Gating: learns how much center vs. context to keep per dimension
        self.gate_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, gene_emb: torch.Tensor, neighbor_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_emb:      [B, dim]    perturbed gene embedding (center)
            neighbor_embs: [B, K, dim] K top PPI neighbor embeddings (context)
        Returns:
            output: [B, dim] gated center+context representation
        """
        # Scaled dot-product attention: gene as query, neighbors as keys/values
        q      = self.q_proj(gene_emb).unsqueeze(1)      # [B, 1, attn_dim]
        k      = self.k_proj(neighbor_embs)               # [B, K, attn_dim]
        scores = (q * k).sum(-1) * self.scale             # [B, K]
        weights = F.softmax(scores, dim=-1)               # [B, K]
        context = (weights.unsqueeze(-1) * neighbor_embs).sum(1)  # [B, dim]

        # Element-wise sigmoid gate: blends center and aggregated neighborhood
        gate   = torch.sigmoid(self.gate_proj(gene_emb)) # [B, dim]
        output = gate * gene_emb + (1.0 - gate) * context
        return output


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
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        input_ids  = tokenized["input_ids"]   # [B, AIDO_GENES] float32

        # Position of the perturbed gene in the AIDO.Cell sequence
        gene_in_vocab  = (input_ids > -1.0).any(dim=1)
        gene_positions = torch.where(
            gene_in_vocab,
            (input_ids > -1.0).float().argmax(dim=1),
            torch.zeros(len(batch), dtype=torch.long),
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
    def __init__(self, batch_size: int = 16, num_workers: int = 4) -> None:
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

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")
        self.train_ds = DEGDataset(train_df)
        self.val_ds   = DEGDataset(val_df)
        self.test_ds  = DEGDataset(test_df)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=make_collate(self.tokenizer))

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------
class HybridNeighborhoodModel(pl.LightningModule):
    """Hybrid: AIDO.Cell-10M (QKV-Muon) + frozen STRING_GNN K=16 Neighborhood Attention.

    Key design decisions:
    - STRING_GNN K=16 neighborhood attention (attn_dim=64, center-context gating):
      Proven to achieve F1=0.4846 in the STRING_GNN lineage (node1-1-1-1-1)
      vs. direct lookup used in sibling node3-1-2 (F1=0.4407)
    - STRING_GNN frozen: avoids fine-tuning risk on small dataset; the neighborhood
      attention module learns to extract the most relevant PPI topology signal
    - AIDO.Cell QKV-only Muon: conservative, proven fine-tuning approach
    - CosineAnnealingLR (no warm restarts): avoids epoch-15 restart that disrupted
      sibling node3-1-2 training (val F1 dropped 0.441 → 0.433)
    - Reduced head (256, dropout=0.4): addresses overfitting on 1388 samples
    """

    def __init__(
        self,
        fusion_layers:   int   = 4,      # last N AIDO.Cell layers to concatenate
        head_hidden:     int   = 256,    # classification head hidden size
        head_dropout:    float = 0.4,    # head dropout rate
        lr_muon:         float = 0.02,   # Muon LR for AIDO QKV weight matrices
        lr_adamw:        float = 2e-4,   # AdamW LR for head + attention params
        weight_decay:    float = 2e-2,   # L2 regularization
        max_epochs:      int   = 80,     # CosineAnnealingLR T_max (matches max training)
        label_smoothing: float = 0.1,    # label smoothing epsilon
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load AIDO.Cell-10M backbone ----
        self.backbone = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        self.backbone = self.backbone.to(torch.bfloat16)
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Enable FlashAttention to avoid 88 GB O(N^2) attention matrix OOM
        self.backbone.config._use_flash_attention_2 = True

        # Share QKV weight tensors between flash_self and regular self attention
        # so that training the regular QKV params also trains the flash path
        for layer in self.backbone.bert.encoder.layer:
            ss = layer.attention.flash_self   # BertSelfFlashAttention
            mm = layer.attention.self          # CellFoundationSelfAttention
            ss.query.weight = mm.query.weight
            ss.key.weight   = mm.key.weight
            ss.value.weight = mm.value.weight
            ss.query.bias   = mm.query.bias
            ss.key.bias     = mm.key.bias
            ss.value.bias   = mm.value.bias

        # Freeze all, then selectively unfreeze QKV weight matrices (not biases)
        # QKV-only Muon: the proven recipe from node3 (F1=0.426) and node3-1-1 (0.4325)
        # Using QKV-only (not QKV+Output) to be conservative and differentiate from sibling1
        for param in self.backbone.parameters():
            param.requires_grad = False

        qkv_weight_suffixes = (
            "attention.self.query.weight",
            "attention.self.key.weight",
            "attention.self.value.weight",
        )
        for name, param in self.backbone.named_parameters():
            if any(name.endswith(s) for s in qkv_weight_suffixes):
                param.requires_grad = True
        # Note: keep QKV in bf16 (required for FlashAttention fast path)

        qkv_count = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total      = sum(p.numel() for p in self.backbone.parameters())
        print(f"[Node3-1-3] AIDO.Cell trainable QKV: {qkv_count:,} / {total:,}")

        # ---- Load STRING_GNN (completely frozen) ----
        self.string_gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
        for param in self.string_gnn.parameters():
            param.requires_grad = False
        self.string_gnn.eval()

        # Load and register STRING graph tensors as buffers (auto-moved to GPU)
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
        self.string_n_nodes = len(node_names)

        # Build top-K neighbor lookup table from graph topology (one-time setup)
        print(f"[Node3-1-3] Building top-{K_NEIGHBORS} PPI neighbor table ...")
        ew_np = graph["edge_weight"].numpy() if ew is not None else None
        nb_table_np = self._build_neighbor_table(
            edge_index=graph["edge_index"].numpy(),
            edge_weight=ew_np,
            N=len(node_names),
            K=K_NEIGHBORS,
        )
        self.register_buffer("neighbor_idx_table",
                              torch.from_numpy(nb_table_np).long())
        print(f"[Node3-1-3] Neighbor table shape: {nb_table_np.shape}")

        # ---- Neighborhood attention with center-context gating ----
        # attn_dim=64: proven best (attn_dim=32 degraded F1 by ~0.01 in node1-1-1-1-1-1)
        self.neighbor_attn = NeighborhoodAttentionGated(dim=HIDDEN_DIM, attn_dim=ATTN_DIM)

        # ---- Classification head ----
        # Input: AIDO concat(fusion_layers × 256) + STRING neighborhood(256)
        #        = 4×256 + 256 = 1280
        # Smaller than sibling node3-1-2 (head_hidden=512): reduces overfitting
        in_dim = HIDDEN_DIM * hp.fusion_layers + HIDDEN_DIM  # 1280
        self.head = nn.Sequential(
            nn.Linear(in_dim, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )

        # Cast head and attention to float32 for stable optimization
        for m in [self.neighbor_attn, self.head]:
            for param in m.parameters():
                param.data = param.data.float()

        # ---- Loss: label-smoothed CE + class weights ----
        # Proven safe: node3 (F1=0.426) and node3-1-1 (F1=0.4325)
        # Label smoothing=0.1: slightly softens targets, reduces overconfidence
        class_weights = get_class_weights()
        self.register_buffer("class_weights", class_weights)

        # State lists for epoch-level aggregation
        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    @staticmethod
    def _build_neighbor_table(
        edge_index:  np.ndarray,           # [2, E] int array
        edge_weight: Optional[np.ndarray], # [E] float array or None
        N: int,
        K: int,
    ) -> np.ndarray:
        """Build a [N, K] table of top-K neighbor indices ordered by edge weight.

        For nodes with fewer than K neighbors, remaining slots are filled with
        self-loops (node → itself), which acts as neutral padding for attention.
        """
        src_nodes = edge_index[0]   # [E]
        dst_nodes = edge_index[1]   # [E]
        E = len(src_nodes)

        # Initialize with self-loops (neutral padding)
        neighbor_table = np.zeros((N, K), dtype=np.int64)
        for i in range(N):
            neighbor_table[i, :] = i

        if edge_weight is None:
            edge_weight = np.ones(E, dtype=np.float32)

        # Sort edges: primary = src ascending, secondary = weight descending
        # np.lexsort sorts by the LAST key first (primary)
        sort_order  = np.lexsort((-edge_weight, src_nodes))
        sorted_src  = src_nodes[sort_order]
        sorted_dst  = dst_nodes[sort_order]

        # np.unique on sorted array gives contiguous group start indices
        unique_src, start_idx, counts = np.unique(
            sorted_src, return_index=True, return_counts=True
        )
        for src, start, cnt in zip(unique_src, start_idx, counts):
            k_actual = min(K, int(cnt))
            neighbor_table[src, :k_actual] = sorted_dst[start:start + k_actual]

        return neighbor_table

    # ---- forward ----
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        gene_positions: torch.Tensor,
        pert_ids:       List[str],
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, AIDO_GENES]  AIDO.Cell tokenized input
            attention_mask: [B, AIDO_GENES]  attention mask
            gene_positions: [B]              position of perturbed gene in AIDO sequence
            pert_ids:       List[str]        Ensembl gene IDs (e.g. ENSG00000012048)
        Returns:
            logits: [B, N_CLASSES, N_GENES]  class logits for all 6640 genes
        """
        B      = input_ids.shape[0]
        device = input_ids.device

        # ---- AIDO.Cell forward: concatenate last fusion_layers hidden states ----
        aido_out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = aido_out.hidden_states  # tuple of len N_AIDO_LAYERS+1

        n = self.hparams.fusion_layers
        layer_embs = []
        for i in range(n):
            hs = hidden_states[-(i + 1)]          # [B, AIDO_GENES+2, 256]
            ge = hs[torch.arange(B, device=device), gene_positions, :].float()  # [B, 256]
            layer_embs.append(ge)
        aido_feat = torch.cat(layer_embs, dim=-1)  # [B, 1024]

        # ---- STRING_GNN forward: frozen, run once per batch ----
        with torch.no_grad():
            string_all_embs = self.string_gnn(
                edge_index=self.edge_index,
                edge_weight=self.edge_weight,
            ).last_hidden_state.float()  # [N_STRING_NODES, 256]

        # Map Ensembl pert_ids to STRING node indices
        # Genes not in STRING graph get index -1 → clamped to 0, then zeroed out
        gene_idx_list   = [self.string_name_to_idx.get(pid, -1) for pid in pert_ids]
        gene_idx_tensor = torch.tensor(gene_idx_list, dtype=torch.long, device=device)  # [B]
        valid_mask      = (gene_idx_tensor >= 0).float().unsqueeze(1)  # [B, 1]
        clamped_idx     = gene_idx_tensor.clamp(min=0)                 # [B]

        # Center embedding (perturbed gene itself)
        gene_embs = string_all_embs[clamped_idx]               # [B, 256]

        # K-neighbor embeddings (top-K by STRING edge confidence)
        nb_idx  = self.neighbor_idx_table[clamped_idx]         # [B, K]
        nb_embs = string_all_embs[nb_idx.view(-1)].view(B, K_NEIGHBORS, HIDDEN_DIM)  # [B, K, 256]

        # Neighborhood attention + center-context gating → [B, 256]
        string_feat = self.neighbor_attn(gene_embs, nb_embs)

        # Zero out features for genes not in STRING graph
        string_feat = string_feat * valid_mask   # [B, 256]

        # Fuse: AIDO.Cell (1024) + STRING neighborhood (256) = 1280
        fused  = torch.cat([aido_feat, string_feat], dim=-1)  # [B, 1280]
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)  # [B, 3, 6640]
        return logits

    # ---- loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G    = logits.shape
        flat_log   = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, C]
        flat_tgt   = targets.reshape(-1)                      # [B*G]
        return F.cross_entropy(
            flat_log, flat_tgt,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ---- Lightning steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"],
                      batch["gene_positions"], batch["pert_id"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True,
                 on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"],
                      batch["gene_positions"], batch["pert_id"])
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

        # Sort + deduplicate (handles DDP padding)
        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device),
                             s_idx[1:] != s_idx[:-1]])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"],
                      batch["gene_positions"], batch["pert_id"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)
        local_idx   = torch.cat(self._test_idx,   0)

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        if self.trainer.is_global_zero:
            # Sort and deduplicate by sample index (handles DDP padding)
            order  = torch.argsort(all_idx)
            s_idx  = all_idx[order]
            s_pred = all_preds[order]
            unique_mask = torch.cat([torch.tensor([True], device=s_idx.device),
                                     s_idx[1:] != s_idx[:-1]])
            s_idx  = s_idx[unique_mask]
            s_pred = s_pred[unique_mask]

            # Look up pert_id and symbol from the test dataset by sample index
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
            print(f"[Node3-1-3] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()

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
        self.print(f"Checkpoint: {trained:,}/{total:,} params ({100*trained/total:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer: Muon for AIDO QKV, AdamW for head + attention ----
    def configure_optimizers(self):
        hp = self.hparams

        # Muon group: AIDO.Cell QKV weight matrices (ndim >= 2, requires_grad)
        qkv_weights = [
            p for name, p in self.backbone.named_parameters()
            if p.requires_grad and p.ndim >= 2
        ]

        # AdamW group: neighborhood attention params + classification head params
        adamw_params = (
            list(self.neighbor_attn.parameters()) +
            list(self.head.parameters())
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

        # CosineAnnealingLR: smooth monotonic decay over max_epochs, no warm restarts.
        # Avoids the epoch-15 restart disruption observed in sibling node3-1-2
        # (val F1 dropped from 0.441 to 0.433 at the SGDR restart point).
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
        description="Node3-1-3: STRING_GNN K=16 Neighborhood Attention + AIDO.Cell-10M + QKV + Muon"
    )
    parser.add_argument("--micro-batch-size",   type=int,   default=16)
    parser.add_argument("--global-batch-size",  type=int,   default=128)
    parser.add_argument("--max-epochs",         type=int,   default=80)
    parser.add_argument("--lr-muon",            type=float, default=0.02)
    parser.add_argument("--lr-adamw",           type=float, default=2e-4)
    parser.add_argument("--weight-decay",       type=float, default=2e-2)
    parser.add_argument("--fusion-layers",      type=int,   default=4)
    parser.add_argument("--head-hidden",        type=int,   default=256)
    parser.add_argument("--head-dropout",       type=float, default=0.4)
    parser.add_argument("--label-smoothing",    type=float, default=0.1)
    parser.add_argument("--num-workers",        type=int,   default=4)
    parser.add_argument("--debug_max_step",     type=int,   default=None)
    parser.add_argument("--fast_dev_run",       action="store_true")
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

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = HybridNeighborhoodModel(
        fusion_layers   = args.fusion_layers,
        head_hidden     = args.head_hidden,
        head_dropout    = args.head_dropout,
        lr_muon         = args.lr_muon,
        lr_adamw        = args.lr_adamw,
        weight_decay    = args.weight_decay,
        max_epochs      = args.max_epochs,
        label_smoothing = args.label_smoothing,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath  = str(output_dir / "checkpoints"),
        filename = "best-{epoch:03d}-{val/f1:.4f}",
        monitor  = "val/f1", mode="max", save_top_k=1,
    )
    # patience=10, min_delta=0.002: proven in both siblings
    # Prevents overfitting while allowing genuine improvement windows
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=10, min_delta=0.002)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    strategy = (
        DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120))
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
        val_check_interval      = (args.val_check_interval
                                   if (args.debug_max_step is None and not fast_dev_run)
                                   else 1.0),
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

    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node3-1-3] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
