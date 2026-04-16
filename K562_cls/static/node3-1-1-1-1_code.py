"""Node1-2 – STRING_GNN (frozen, cond_emb-conditioned) + AIDO.Cell-10M (QKV-only) Hybrid.

Improvements over parent node (node3-1-1-1, F1=0.3989):

1. RESTORE HEAD CAPACITY: head_hidden=512 (from 256). The parent's head_hidden=256 + dropout=0.5
   + wd_head=0.05 was catastrophically over-regularized. Restoring to 512 matches the grandparent
   node3-1-1 (F1=0.4325) which used head_hidden=512 and achieved much better results.

2. FREEZE STRING_GNN (use as fixed feature extractor): Instead of full fine-tuning 5.43M GNN
   parameters on only 1,388 samples (which caused overfitting in the parent), we use STRING_GNN
   as a frozen pretrained encoder. This provides the same PPI structural knowledge without the
   overfitting risk. The frozen embeddings are still highly informative (node1-1 achieved F1=0.453
   using frozen STRING_GNN). Zero GNN training overhead also speeds up training significantly.

3. PERTURBATION-AWARE GNN CONDITIONING via cond_emb: The STRING_GNN skill documents a `cond_emb`
   input [N, 256] that applies additive per-node conditioning before message passing. We use the
   AIDO.Cell gene embedding (256-dim extracted from the last layer at the perturbed gene position)
   as conditioning signal for STRING_GNN. This creates a unique perturbation-specific graph
   embedding where the PPI topology is modulated by the expression context of the perturbed gene.
   This integration of AIDO.Cell features into STRING_GNN's message passing is novel — not tried
   in any previous node.

4. BALANCED REGULARIZATION: dropout=0.35 (from 0.5), wd_head=0.02 (from 0.05). The feedback
   from node3-1-1-1 clearly identified dropout=0.5 + wd_head=0.05 as the primary failure cause
   (over-regularization → underfitting). 0.35 provides meaningful regularization without blocking
   gradient flow. 0.02 is standard for classification heads.

5. TWO-LAYER HEAD with residual: Linear(1280→512) → LN → GELU → Dropout(0.35) →
   Linear(512→256) → LN → GELU → Dropout(0.2) → Linear(256→N_CLASSES*N_GENES).
   A second hidden layer provides richer feature transformation for the 6,640-gene task
   while moderate per-layer dropout provides regularization without bottlenecking.

6. REDUCELRONPLATEAU instead of CosineAnnealingWarmRestarts: All cosine warm restart nodes
   (T_0=25 or T_0=50) failed to fire warm restarts due to early stopping. ReduceLROnPlateau
   is adaptive — it responds to actual validation F1 plateaus rather than a fixed schedule.
   This is more appropriate for the variable convergence dynamics observed across nodes.

7. SEPARATE OPTIMIZER for frozen GNN: Since STRING_GNN is frozen, it has no optimizer group.
   Only AIDO.Cell QKV matrices (Muon) and head (AdamW) are optimized. Simpler optimizer
   configuration eliminates the GNN-optimizer interaction complexity from the parent.

Memory from nodes used:
- node3-1-1-1 feedback: "over-regularization is primary failure, restore head_hidden=512,
  dropout=0.3, use STRING_GNN as frozen (non-fine-tuned) embeddings"
- node1-1 (F1=0.453): "STRING_GNN frozen+bilinear very effective, dropout=0.4 mild overfitting"
- node3-1-1 (F1=0.4325): "head_hidden=512, dropout=0.3 worked well with AIDO.Cell"
- node4 (F1=0.4585): "scFoundation+STRING_GNN fusion is best; cosine annealing + higher wd"
- STRING_GNN skill: "cond_emb=[N,256] applies additive per-node conditioning before message passing"

Architecture:
  Step 1: AIDO.Cell-10M → extract last-layer hidden state at gene position → [B, 256] float32
  Step 2: STRING_GNN(edge_index, edge_weight, cond_emb=aido_gene_emb) → [18870, 256] float32
          (perturb the PPI graph with AIDO.Cell's expression context for the gene)
  Step 3: Index STRING_GNN output at pert_gene position → [B, 256]
  Step 4: AIDO.Cell-10M 4-layer concat at gene position → [B, 1024]
  Step 5: fused = cat([gnn_emb, cell_emb]) → [B, 1280]
  Step 6: head: Linear(1280→512) → LN → GELU → Dropout(0.35) →
                Linear(512→256) → LN → GELU → Dropout(0.20) →
                Linear(256→19920) → [B, 3, 6640]

Optimizer:
  Muon: AIDO.Cell QKV weight matrices (lr=0.02, wd=0.01) — proven effective across all nodes
  AdamW: head params (lr=2e-4, wd=0.02) — balanced L2, not too aggressive
  ReduceLROnPlateau: adaptive decay based on val/f1 plateau
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
N_GENES        = 6640
N_CLASSES      = 3
AIDO_GENES     = 19264
AIDO_CELL_DIR  = "/home/Models/AIDO.Cell-10M"
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
CELL_HIDDEN    = 256    # AIDO.Cell-10M hidden size
GNN_HIDDEN     = 256    # STRING_GNN output dim
FUSION_LAYERS  = 4      # AIDO.Cell layers to concat (proven in node3 and node3-1-1)

# Class label frequencies: down (class 0), neutral (class 1), up (class 2)
# Labels remapped from {-1,0,1} to {0,1,2}
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency class weights to handle 92.5% neutral imbalance."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def load_gnn_node_mapping() -> Dict[str, int]:
    """Build pert_id → STRING_GNN node_index dict from node_names.json.
    node_names.json maps index → Ensembl ID; may include version suffix (ENSGXXXXX.1).
    We strip version suffixes for robust matching.
    """
    node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    # Strip version suffix (e.g., ENSG00000012048.3 → ENSG00000012048)
    return {name.split(".")[0]: i for i, name in enumerate(node_names)}


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute per-gene macro F1 matching calc_metric.py logic.

    Args:
        preds: [N, 3, G] softmax probabilities
        targets: [N, G] integer class labels (0=down, 1=neutral, 2=up)
    Returns:
        Mean per-gene macro F1 over all G genes.
    """
    y_hat       = preds.argmax(dim=1)  # [N, G]
    G           = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)
    for c in range(3):
        is_true = (targets == c)
        is_pred = (y_hat == c)
        present = is_true.any(dim=0).float()
        tp      = (is_pred & is_true).float().sum(0)
        fp      = (is_pred & ~is_true).float().sum(0)
        fn      = (~is_pred & is_true).float().sum(0)
        prec    = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec     = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c    = torch.where(prec + rec > 0, 2*prec*rec/(prec+rec+1e-8), torch.zeros_like(prec))
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# Dataset / DataModule
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame, gnn_node_map: Dict[str, int]) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()

        has_label = "label" in df.columns and df["label"].notna().all()
        self.labels: Optional[List] = (
            [torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
             for row in df["label"].tolist()]
            if has_label else None
        )

        # Map pert_id → STRING_GNN node index; -1 if not in graph
        self.gnn_indices: List[int] = [
            gnn_node_map.get(pid.split(".")[0], -1)
            for pid in self.pert_ids
        ]

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "sample_idx":  idx,
            "pert_id":     self.pert_ids[idx],
            "symbol":      self.symbols[idx],
            "gnn_node_idx": self.gnn_indices[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def make_collate(tokenizer):
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]

        # Tokenize perturbation as single gene with expression=1.0
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        input_ids  = tokenized["input_ids"]  # [B, 19264] float32

        # Locate the perturbed gene's position in the 19264-gene input
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
            "gnn_node_idx":   torch.tensor([b["gnn_node_idx"] for b in batch], dtype=torch.long),
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
            AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)

        gnn_node_map = load_gnn_node_mapping()

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df, gnn_node_map)
        self.val_ds   = DEGDataset(val_df,   gnn_node_map)
        self.test_ds  = DEGDataset(test_df,  gnn_node_map)

        # Log STRING_GNN coverage
        n_train_in = sum(1 for x in self.train_ds.gnn_indices if x >= 0)
        n_val_in   = sum(1 for x in self.val_ds.gnn_indices   if x >= 0)
        n_test_in  = sum(1 for x in self.test_ds.gnn_indices  if x >= 0)
        print(f"[DataModule] STRING_GNN coverage: "
              f"train {n_train_in}/{len(self.train_ds)}, "
              f"val {n_val_in}/{len(self.val_ds)}, "
              f"test {n_test_in}/{len(self.test_ds)}")

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=make_collate(self.tokenizer),
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class StringGNNCondAIDOFusion(pl.LightningModule):
    """STRING_GNN (frozen, cond_emb-conditioned) + AIDO.Cell-10M (QKV-only) hybrid.

    Novel fusion design:
    1. Run AIDO.Cell on the perturbed gene to get expression-context embedding (256-dim)
    2. Use that AIDO.Cell embedding as cond_emb for STRING_GNN — this conditions the
       PPI message passing on the expression context of the perturbed gene, producing
       a perturbation-aware graph embedding rather than a static structural embedding.
    3. Extract STRING_GNN's conditioned node embedding for the perturbed gene
    4. Concatenate [GNN-cond(256), AIDO-4layer(1024)] → 1280-dim
    5. Two-layer classification head with moderate regularization

    Key improvements over parent (node3-1-1-1, F1=0.3989):
    - STRING_GNN frozen (eliminates 5.43M overfitting risk on 1,388 samples)
    - cond_emb conditioning: AIDO.Cell gene context modulates PPI topology
    - head_hidden=512 (from 256) restoring grandparent's proven capacity
    - dropout=0.35 (from 0.5) balanced regularization
    - wd_head=0.02 (from 0.05) reasonable L2 penalty
    - Two-layer head for richer feature extraction
    - ReduceLROnPlateau for adaptive scheduling (cosine restarts never fired)
    """

    def __init__(
        self,
        fusion_layers:    int   = 4,      # AIDO.Cell layers to concatenate
        head_hidden1:     int   = 512,    # restored from parent's 256; matches grandparent
        head_hidden2:     int   = 256,    # second hidden layer for richer head
        head_dropout1:    float = 0.35,   # reduced from 0.5; balanced regularization
        head_dropout2:    float = 0.20,   # lighter on deeper layer
        lr_muon:          float = 0.02,   # Muon lr for AIDO.Cell QKV square matrices
        lr_adamw:         float = 2e-4,   # AdamW lr for head
        wd_backbone:      float = 1e-2,   # weight decay for AIDO.Cell QKV
        wd_head:          float = 2e-2,   # reduced from 0.05; standard for classification head
        label_smoothing:  float = 0.1,
        lr_patience:      int   = 5,      # ReduceLROnPlateau patience
        lr_factor:        float = 0.5,    # ReduceLROnPlateau factor
        lr_min:           float = 1e-6,   # minimum learning rate
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---------------------------------------------------------------
        # 1. AIDO.Cell-10M backbone (QKV-only fine-tuning)
        # ---------------------------------------------------------------
        self.backbone = AutoModel.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        self.backbone = self.backbone.to(torch.bfloat16)
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.backbone.config._use_flash_attention_2 = True

        # Share QKV weights between flash_self and self (for consistent gradient flow)
        for layer in self.backbone.bert.encoder.layer:
            ss = layer.attention.flash_self  # BertSelfFlashAttention
            mm = layer.attention.self        # CellFoundationSelfAttention
            ss.query.weight = mm.query.weight
            ss.key.weight   = mm.key.weight
            ss.value.weight = mm.value.weight
            ss.query.bias   = mm.query.bias
            ss.key.bias     = mm.key.bias
            ss.value.bias   = mm.value.bias

        # Freeze all, then unfreeze QKV weights only
        for param in self.backbone.parameters():
            param.requires_grad = False
        qkv_patterns = (
            "attention.self.query.weight",
            "attention.self.key.weight",
            "attention.self.value.weight",
        )
        for name, param in self.backbone.named_parameters():
            if any(name.endswith(p) for p in qkv_patterns):
                param.requires_grad = True

        # NOTE: Do NOT cast trainable QKV params to float32 here.
        # Casting QKV weights to float32 while keeping bf16 input activations
        # causes a dtype mismatch in the AIDO.Cell attention matmul (float32 QKV
        # output interacting with bf16 K/V in the attention score computation),
        # triggering OOM during backward pass. Muon optimizer works correctly
        # with bf16 parameters — the in-place updates (p.mul_, p.add_) execute
        # in bf16, which is numerically stable for this optimizer.

        qkv_count = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total      = sum(p.numel() for p in self.backbone.parameters())
        print(f"[Node] AIDO.Cell-10M trainable: {qkv_count:,} / {total:,} params")

        # ---------------------------------------------------------------
        # 2. STRING_GNN backbone (FROZEN — used as pretrained feature extractor)
        #    With cond_emb conditioning from AIDO.Cell embedding for perturbation-aware PPI
        # ---------------------------------------------------------------
        self.gnn = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        self.gnn = self.gnn.float()  # float32 for stable GCN computation

        # FREEZE all GNN parameters — no fine-tuning on 1,388 samples
        # The pretrained PPI topology knowledge is preserved without overfitting risk
        for param in self.gnn.parameters():
            param.requires_grad = False

        gnn_total = sum(p.numel() for p in self.gnn.parameters())
        print(f"[Node] STRING_GNN: {gnn_total:,} params (FROZEN — no fine-tuning)")

        # Load graph tensors as non-persistent buffers (auto device move, not checkpointed)
        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        self.register_buffer("gnn_edge_index",  graph["edge_index"].long(),  persistent=False)
        _ew = graph.get("edge_weight")
        if _ew is not None:
            self.register_buffer("gnn_edge_weight", _ew.float(), persistent=False)
        else:
            n_edges = graph["edge_index"].shape[1]
            self.register_buffer("gnn_edge_weight",
                                  torch.ones(n_edges, dtype=torch.float32), persistent=False)

        # Number of STRING_GNN nodes (18,870) for cond_emb
        self._gnn_n_nodes = graph["edge_index"].max().item() + 1
        print(f"[Node] STRING_GNN graph: {self._gnn_n_nodes} nodes, "
              f"{graph['edge_index'].shape[1]} edges")

        # ---------------------------------------------------------------
        # 3. Fusion head (two-layer MLP with balanced regularization)
        # ---------------------------------------------------------------
        # Input: cat([STRING_GNN 256-dim cond, AIDO.Cell 4-layer concat 1024-dim]) = 1280-dim
        in_dim = GNN_HIDDEN + hp.fusion_layers * CELL_HIDDEN  # 256 + 4*256 = 1280
        self.head = nn.Sequential(
            nn.Linear(in_dim, hp.head_hidden1),
            nn.LayerNorm(hp.head_hidden1),
            nn.GELU(),
            nn.Dropout(hp.head_dropout1),
            nn.Linear(hp.head_hidden1, hp.head_hidden2),
            nn.LayerNorm(hp.head_hidden2),
            nn.GELU(),
            nn.Dropout(hp.head_dropout2),
            nn.Linear(hp.head_hidden2, N_CLASSES * N_GENES),
        )
        self.head = self.head.float()

        head_count = sum(p.numel() for p in self.head.parameters())
        print(f"[Node] Head params: {head_count:,} "
              f"(in={in_dim}, h1={hp.head_hidden1}, h2={hp.head_hidden2})")

        # ---------------------------------------------------------------
        # 4. Loss configuration
        # ---------------------------------------------------------------
        class_weights = get_class_weights()
        self.register_buffer("class_weights", class_weights)
        self.label_smoothing = hp.label_smoothing

        # Accumulators for multi-GPU metric computation
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_tgts:  List[torch.Tensor] = []
        self._test_meta:  List[Tuple]        = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, 19264] float32
        attention_mask: torch.Tensor,   # [B, 19264]
        gene_positions: torch.Tensor,   # [B] long
        gnn_node_idx:   torch.Tensor,   # [B] long (-1 if not in graph)
    ) -> torch.Tensor:
        B      = input_ids.shape[0]
        device = input_ids.device

        # ---- AIDO.Cell: extract 4-layer concat + last-layer gene embedding ----
        cell_out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # hidden_states: tuple of (N_LAYERS+1) tensors, each [B, 19266, 256]
        n = self.hparams.fusion_layers
        layer_embs = []
        for i in range(n):
            hs = cell_out.hidden_states[-(i + 1)]  # [B, 19266, 256] bfloat16
            ge = hs[torch.arange(B, device=device), gene_positions, :].float()  # [B, 256]
            layer_embs.append(ge)

        pert_cell_embs = torch.cat(layer_embs, dim=1)  # [B, 1024] float32

        # Get last-layer gene embedding as cond_emb conditioning for STRING_GNN
        # Use hidden_states[-1] (last transformer layer) at gene position
        last_hs = cell_out.hidden_states[-1]  # [B, 19266, 256] bfloat16
        pert_gene_emb = last_hs[torch.arange(B, device=device), gene_positions, :].float()  # [B, 256]

        # ---- STRING_GNN: conditioned by AIDO.Cell gene embedding ----
        # Build cond_emb: [N_GNN_NODES, 256] with AIDO.Cell embedding at perturbed gene position
        # For genes not in graph: use zero conditioning (no effect on message passing)
        with torch.no_grad():  # GNN is frozen
            cond_emb = torch.zeros(
                self._gnn_n_nodes, GNN_HIDDEN, device=device, dtype=torch.float32
            )
            valid_mask = gnn_node_idx >= 0  # [B] bool
            if valid_mask.any():
                # Use scatter to handle potentially duplicate gnn_node_idx entries safely
                valid_gnn_idx  = gnn_node_idx[valid_mask]   # [n_valid]
                valid_gene_emb = pert_gene_emb[valid_mask]  # [n_valid, 256]
                cond_emb.index_put_((valid_gnn_idx,), valid_gene_emb, accumulate=False)

            gnn_out   = self.gnn(
                edge_index=self.gnn_edge_index,
                edge_weight=self.gnn_edge_weight,
                cond_emb=cond_emb,
            )
            node_embs = gnn_out.last_hidden_state  # [18870, 256] float32

        # Index by perturbed gene; zero for genes not in STRING graph
        pert_gnn_embs = torch.zeros(B, GNN_HIDDEN, device=device, dtype=torch.float32)
        if valid_mask.any():
            pert_gnn_embs[valid_mask] = node_embs[gnn_node_idx[valid_mask]].float()

        # ---- Concatenation fusion ----
        fused = torch.cat([pert_gnn_embs, pert_cell_embs], dim=1)  # [B, 1280]

        # ---- Classification head ----
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        return logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G    = logits.shape
        flat_logits  = logits.permute(0, 2, 1).reshape(-1, C)
        flat_targets = targets.reshape(-1)
        return F.cross_entropy(
            flat_logits, flat_targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["gnn_node_idx"],
        )
        loss = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["gnn_node_idx"],
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

        # De-duplicate (DDP may produce duplicate samples)
        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["gnn_node_idx"],
        )
        probs = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        for i, (pid, sym) in enumerate(zip(batch["pert_id"], batch["symbol"])):
            self._test_meta.append((pid, sym, batch["sample_idx"][i].item()))
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)
            self._test_tgts.append(batch["labels"].detach())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds    = torch.cat(self._test_preds, 0)
        all_preds      = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        local_idx_list = torch.tensor(
            [m[2] for m in self._test_meta], dtype=torch.long, device=all_preds.device
        )
        all_idx = self.all_gather(local_idx_list).view(-1)

        # Compute test F1 if targets are available
        if self._test_tgts:
            local_tgts = torch.cat(self._test_tgts, 0)
            all_tgts   = self.all_gather(local_tgts).view(-1, N_GENES)
            order   = torch.argsort(all_idx)
            s_idx   = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
            mask    = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
            test_f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
            self.log("test/f1", test_f1, prog_bar=True, sync_dist=True)

        # Gather metadata from all ranks for prediction file
        world_size = self.trainer.world_size if hasattr(self.trainer, "world_size") else 1
        all_meta_flat: List[Tuple] = []
        if world_size > 1:
            gathered_meta: List[List] = [None] * world_size
            torch.distributed.all_gather_object(gathered_meta, list(self._test_meta))
            for meta_list in gathered_meta:
                all_meta_flat.extend(meta_list)
        else:
            all_meta_flat = list(self._test_meta)

        if self.trainer.is_global_zero:
            meta_dict: Dict[int, Tuple] = {m[2]: m for m in all_meta_flat}
            n_samples = all_preds.shape[0]
            rows = []
            for i in range(n_samples):
                idx_val = all_idx[i].item()
                pid, sym = meta_dict.get(idx_val, (f"unknown_{idx_val}", f"unknown_{idx_val}"))[:2]
                rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": json.dumps(all_preds[i].float().cpu().numpy().tolist()),
                })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_tgts.clear()
        self._test_meta.clear()

    # ------------------------------------------------------------------
    # Checkpoint: save only trainable parameters + buffers
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
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Checkpoint: {trained}/{total} params ({100*trained/total:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Optimizer: Muon for AIDO.Cell QKV, AdamW for head
    # ReduceLROnPlateau for adaptive LR scheduling (cosine restarts never fire in practice)
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams

        # Muon group: AIDO.Cell QKV weight matrices (all 256×256 square — ideal for Muon)
        aido_qkv_matrices = [
            p for _, p in self.backbone.named_parameters()
            if p.requires_grad and p.ndim >= 2
        ]
        # Handle AIDO.Cell trainable 1D params (e.g., biases)
        aido_biases = [
            p for _, p in self.backbone.named_parameters()
            if p.requires_grad and p.ndim < 2
        ]

        head_params = list(self.head.parameters())
        # STRING_GNN is frozen — no optimizer group needed

        param_groups = [
            # Muon: AIDO.Cell QKV weight matrices
            dict(
                params       = aido_qkv_matrices,
                use_muon     = True,
                lr           = hp.lr_muon,
                weight_decay = hp.wd_backbone,
                momentum     = 0.95,
            ),
            # AdamW: head + any AIDO.Cell biases — balanced regularization
            dict(
                params       = head_params + aido_biases,
                use_muon     = False,
                lr           = hp.lr_adamw,
                betas        = (0.9, 0.95),
                weight_decay = hp.wd_head,
            ),
        ]

        use_distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        opt_cls   = MuonWithAuxAdam if use_distributed else SingleDeviceMuonWithAuxAdam
        optimizer = opt_cls(param_groups)

        # ReduceLROnPlateau: adaptive decay based on val/f1 improvement.
        # More robust than CosineAnnealingWarmRestarts which never fires warm restarts
        # in practice (early stopping triggers before T_0 completes).
        # patience=5: reduce LR after 5 epochs without improvement in val/f1
        # factor=0.5: halve the LR each time
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode     = "max",       # maximize val/f1
            factor   = hp.lr_factor,
            patience = hp.lr_patience,
            min_lr   = hp.lr_min,
            verbose  = True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor":   "val/f1",
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
        description="Node1-2 – STRING_GNN (frozen, cond_emb) + AIDO.Cell-10M QKV Hybrid"
    )
    parser.add_argument("--micro-batch-size",  type=int,   default=8)
    parser.add_argument("--global-batch-size", type=int,   default=128)
    parser.add_argument("--max-epochs",        type=int,   default=100)
    parser.add_argument("--fusion-layers",     type=int,   default=4)
    parser.add_argument("--head-hidden1",      type=int,   default=512)
    parser.add_argument("--head-hidden2",      type=int,   default=256)
    parser.add_argument("--head-dropout1",     type=float, default=0.35)
    parser.add_argument("--head-dropout2",     type=float, default=0.20)
    parser.add_argument("--lr-muon",           type=float, default=0.02)
    parser.add_argument("--lr-adamw",          type=float, default=2e-4)
    parser.add_argument("--wd-backbone",       type=float, default=1e-2)
    parser.add_argument("--wd-head",           type=float, default=2e-2)
    parser.add_argument("--label-smoothing",   type=float, default=0.1)
    parser.add_argument("--lr-patience",       type=int,   default=5)
    parser.add_argument("--lr-factor",         type=float, default=0.5)
    parser.add_argument("--lr-min",            type=float, default=1e-6)
    parser.add_argument("--es-patience",       type=int,   default=15)
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--debug_max_step",    type=int,   default=None)
    parser.add_argument("--fast_dev_run",      action="store_true")
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
    model = StringGNNCondAIDOFusion(
        fusion_layers   = args.fusion_layers,
        head_hidden1    = args.head_hidden1,
        head_hidden2    = args.head_hidden2,
        head_dropout1   = args.head_dropout1,
        head_dropout2   = args.head_dropout2,
        lr_muon         = args.lr_muon,
        lr_adamw        = args.lr_adamw,
        wd_backbone     = args.wd_backbone,
        wd_head         = args.wd_head,
        label_smoothing = args.label_smoothing,
        lr_patience     = args.lr_patience,
        lr_factor       = args.lr_factor,
        lr_min          = args.lr_min,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath  = str(output_dir / "checkpoints"),
        filename = "best-{epoch:03d}-{val/f1:.4f}",
        monitor  = "val/f1",
        mode     = "max",
        save_top_k = 1,
    )
    # patience=15: reasonable patience to avoid over-early stopping
    # The prior over-regularization caused convergence at epoch 14 — allow sufficient time
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=args.es_patience, min_delta=0.001)
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
        val_check_interval      = args.val_check_interval if (
            args.debug_max_step is None and not fast_dev_run
        ) else 1.0,
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
    print(f"[Node] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
