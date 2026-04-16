"""Node 2-1-2-1-1-1-1-1-1-1: Extended Training for Third Warm Restart + Reduced Dropout + Label Smoothing

KEY CHANGES over parent node2-1-2-1-1-1-1-1-1 (Test F1=0.5283):

  1. EXTEND TRAINING TO CAPTURE THIRD WARM RESTART (HIGH PRIORITY)
     - max_epochs: 500 → 600 (allows third warm restart cycle at epoch 240 to complete)
     - patience: 100 → 130 (training was stopping at epoch 238, just 12 epochs before
       third restart at ~epoch 250; with patience=130, the model can explore the full
       third cycle without premature stopping)
     - T_0=80, T_mult=2: restarts at epochs 80, 240, 560(>600) — third cycle covers 240-560
     - Parent feedback explicitly identified this as the #1 bottleneck: "training stopped
       at epoch 238, just 12 epochs before the expected third restart at epoch ~250"
     - Expected impact: +0.003–0.008 F1 from additional landscape escape

  2. REDUCE head_dropout: 0.35 → 0.30 (HIGH PRIORITY)
     - Tree-best (node3-1-1-1-1-2-1-1-1, F1=0.5283) uses dropout=0.30
     - Parent (this node's parent) used 0.35 and tied tree-best; further reduction may
       allow slightly better training fit without over-regularizing
     - With per_gene_bias removed and weight_decay=2e-3, the effective regularization
       budget can accommodate lower dropout
     - Expected impact: +0.002–0.005 F1 by allowing better training convergence

  3. ADD LABEL SMOOTHING ε=0.05 (MEDIUM PRIORITY)
     - Train/val loss ratio at parent's final epoch: 38.8× (0.033 train vs 1.287 val)
     - This extreme ratio indicates over-confident logits — model assigns near-zero
       probability to all non-argmax classes, which degrades calibration
     - Label smoothing ε=0.05 adds small uniform probability to non-target classes,
       preventing the model from becoming arbitrarily overconfident
     - Uses PyTorch's built-in F.cross_entropy(label_smoothing=0.05) with class weights
     - Applied to hard-label loss only (WCE); soft Mixup loss unchanged
     - Expected impact: +0.002–0.004 F1 from improved calibration and generalization

  4. INCREASE ENSEMBLE to top-9 checkpoints (MEDIUM PRIORITY)
     - With max_epochs=600 and three full CosineWarmRestarts cycles, more diverse
       checkpoints are available spanning epochs 80–560
     - Ensemble diversity increases with cycle span → better variance reduction
     - Parent used top-7 and achieved 0.5283; top-9 with more diverse checkpoints
       (cycle 2 peak + cycle 3 peak) should compound the benefit
     - Expected impact: +0.001–0.002 F1 from additional variance reduction

  5. All proven components UNCHANGED from parent:
     - Frozen ESM2-3B (2560-dim, fully frozen)
     - TwoStepESM2Proj(2560→512→256) — proven to reduce information bottleneck
     - STRING_GNN 4-layer fine-tuning (mps.4-7, tiered LR near=5e-5, far=3e-5)
     - GatedFusion(FUSION_DIM=512)
     - 3-block PreNorm MLP structure (h=384, inner=768)
     - No per_gene_bias (removed to reduce overfitting)
     - MuonWithAuxAdam optimizer (Muon lr=0.01, AdamW lr=3e-4)
     - weight_decay=2e-3
     - Manifold Mixup (prob=0.75, alpha=0.3) — unchanged (already at tree-best level)
     - WCE loss with correct inverse-frequency class weights [0.0477, 0.9282, 0.0241]
     - micro_batch_size=4, global_batch_size=32
     - T_0=80, T_mult=2, warmup=10

Memory rationale:
  - parent node2-1-2-1-1-1-1-1-1 (F1=0.5283): feedback identifies "training stopped
    12 epochs before third warm restart" as the #1 bottleneck; reduce dropout to 0.30;
    add label smoothing for 38.8× loss ratio
  - node3-1-1-1-1-2-1-1-1 (tree-best, F1=0.5283): T_0=80, patience=120, dropout=0.30
  - node3-1-1-1-1-2-1 (ESM2-3B, F1=0.5243): feedback confirms 10:1 compression bottleneck
    resolved by TwoStepProj (preserved from parent)

Auxiliary data dependencies:
  - Protein FASTA: /home/data/genome/hg38_gencode_protein.fa (genomic-data-skill)
  - STRING_GNN:    /home/Models/STRING_GNN (string-gnn-model-skill)
  - ESM2-3B:       facebook/esm2_t36_3B_UR50D (esm2-protein-model-skill)
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, EsmForMaskedLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ESM2_MODEL = "facebook/esm2_t36_3B_UR50D"
PROTEIN_FASTA = "/home/data/genome/hg38_gencode_protein.fa"
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")

N_GENES = 6640
N_CLASSES = 3
MAX_SEQ_LEN = 512        # ESM2 max context
ESM2_DIM = 2560          # ESM2-3B hidden size
STRING_DIM = 256         # STRING_GNN output dimension
FUSION_DIM = 512         # Fusion dimension (proven in tree-best)
HEAD_HIDDEN = 384        # PreNorm MLP hidden dimension (tree-best proven capacity)
FALLBACK_SEQ = "M"       # Minimal placeholder if ENSG not in FASTA
STRING_FALLBACK_IDX = 18870  # Zero-row index for genes absent from STRING


# ---------------------------------------------------------------------------
# Protein sequence lookup helpers
# ---------------------------------------------------------------------------
def _build_ensg_to_seq(fasta_path: str) -> Dict[str, str]:
    """Parse hg38_gencode_protein.fa → ENSG (no version) → longest protein sequence."""
    ensg2seq: Dict[str, str] = {}
    current_ensg: Optional[str] = None
    current_seq_parts: List[str] = []

    def _flush() -> None:
        if current_ensg and current_seq_parts:
            seq = "".join(current_seq_parts)
            if current_ensg not in ensg2seq or len(seq) > len(ensg2seq[current_ensg]):
                ensg2seq[current_ensg] = seq

    with open(fasta_path, "r") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                _flush()
                current_seq_parts = []
                current_ensg = None
                fields = line[1:].split("|")
                if len(fields) >= 3:
                    current_ensg = fields[2].split(".")[0]
            else:
                current_seq_parts.append(line)
    _flush()
    return ensg2seq


_ENSG2SEQ_CACHE: Optional[Dict[str, str]] = None


def get_ensg2seq() -> Dict[str, str]:
    global _ENSG2SEQ_CACHE
    if _ENSG2SEQ_CACHE is None:
        _ENSG2SEQ_CACHE = _build_ensg_to_seq(PROTEIN_FASTA)
    return _ENSG2SEQ_CACHE


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbProteinDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        ensg2seq: Dict[str, str],
        ensg_to_string_idx: Dict[str, int],
        n_string_nodes: int = 18870,
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.n_string_nodes = n_string_nodes

        # Protein sequences
        self.sequences: List[str] = []
        for pid in self.pert_ids:
            ensg = pid.split(".")[0]
            self.sequences.append(ensg2seq.get(ensg, FALLBACK_SEQ))

        # STRING_GNN node indices
        self.string_idxs: List[int] = []
        for pid in self.pert_ids:
            ensg = pid.split(".")[0]
            self.string_idxs.append(ensg_to_string_idx.get(ensg, STRING_FALLBACK_IDX))

        if "label" in df.columns:
            labels = np.array([json.loads(x) for x in df["label"].tolist()], dtype=np.int64)
            self.labels: Optional[torch.Tensor] = torch.tensor(labels + 1, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "seq": self.sequences[idx],
            "string_idx": self.string_idxs[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]  # [6640]
        return item


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class PerturbDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        micro_batch_size: int = 4,
        num_workers: int = 4,
        max_seq_len: int = MAX_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len

        self.tokenizer = None
        self.ensg2seq: Optional[Dict[str, str]] = None
        self.ensg_to_string_idx: Optional[Dict[str, int]] = None
        self.train_ds = self.val_ds = self.test_ds = None
        self.n_string_nodes = 18870

    def setup(self, stage: str = "fit") -> None:
        # Tokenizer: rank-0 downloads first, all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(ESM2_MODEL, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL, trust_remote_code=True)

        # Protein FASTA
        self.ensg2seq = get_ensg2seq()

        # STRING node names: ENSG_ID → STRING node index
        node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        self.n_string_nodes = len(node_names)
        self.ensg_to_string_idx = {name: i for i, name in enumerate(node_names)}

        # Datasets
        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbProteinDataset(train_df, self.ensg2seq, self.ensg_to_string_idx, self.n_string_nodes)
        self.val_ds = PerturbProteinDataset(val_df, self.ensg2seq, self.ensg_to_string_idx, self.n_string_nodes)
        self.test_ds = PerturbProteinDataset(test_df, self.ensg2seq, self.ensg_to_string_idx, self.n_string_nodes)

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        seqs = [item["seq"] for item in batch]
        tokenized = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        result = {
            "idx": torch.tensor([item["idx"] for item in batch], dtype=torch.long),
            "pert_id": [item["pert_id"] for item in batch],
            "symbol": [item["symbol"] for item in batch],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "string_idx": torch.tensor([item["string_idx"] for item in batch], dtype=torch.long),
        }
        if "label" in batch[0]:
            result["label"] = torch.stack([item["label"] for item in batch])
        return result

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )


# ---------------------------------------------------------------------------
# Gated Fusion Module (FUSION_DIM=512)
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Gated fusion of protein and STRING_GNN embeddings.

    Projects protein (ESM2-3B 2560-dim) and STRING (256-dim) to FUSION_DIM=512.
    A gate vector learned from the concatenation selects between the two modalities.
    Output: [B, FUSION_DIM] fused representation.
    """

    def __init__(
        self,
        protein_dim: int = ESM2_DIM,
        string_dim: int = STRING_DIM,
        fusion_dim: int = FUSION_DIM,
    ) -> None:
        super().__init__()
        self.protein_proj = nn.Linear(protein_dim, fusion_dim, bias=True)
        self.protein_norm = nn.LayerNorm(fusion_dim)
        self.string_proj = nn.Linear(string_dim, fusion_dim, bias=True)
        self.string_norm = nn.LayerNorm(fusion_dim)
        self.gate_fc = nn.Linear(fusion_dim * 2, fusion_dim, bias=True)
        nn.init.zeros_(self.gate_fc.bias)

    def forward(self, protein_emb: torch.Tensor, string_emb: torch.Tensor) -> torch.Tensor:
        p = self.protein_norm(self.protein_proj(protein_emb.float()))   # [B, FUSION_DIM]
        s = self.string_norm(self.string_proj(string_emb.float()))       # [B, FUSION_DIM]
        concat = torch.cat([p, s], dim=-1)                               # [B, 2*FUSION_DIM]
        gate = torch.sigmoid(self.gate_fc(concat))                       # [B, FUSION_DIM]
        return gate * p + (1.0 - gate) * s                               # [B, FUSION_DIM]


# ---------------------------------------------------------------------------
# PreNorm Residual Block
# ---------------------------------------------------------------------------
class PreNormResBlock(nn.Module):
    """Pre-normalization residual block for MLP.

    Structure: LN → Linear → GELU → Dropout → Linear → + residual

    KEY CHANGE (vs parent): head_dropout reduced from 0.35 → 0.30
    Tree-best (node3-1-1-1-1-2-1-1-1, F1=0.5283) uses dropout=0.30.
    Parent used 0.35 and tied tree-best; further reduction may allow better fit.
    """

    def __init__(self, hidden_dim: int, inner_dim: int, dropout: float = 0.30) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, inner_dim, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(inner_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x + residual


# ---------------------------------------------------------------------------
# Two-Step ESM2 Projection (from parent — proven to reduce bottleneck)
# ---------------------------------------------------------------------------
class TwoStepESM2Proj(nn.Module):
    """Two-step projection from ESM2-3B space (2560-dim) to STRING space (256-dim).

    KEY INNOVATION (inherited from parent): Reduces 10:1 compression bottleneck.
    Parent's parent used Linear(2560→256) — a single aggressive 10:1 projection.
    This module uses Linear(2560→512) → GELU → Linear(512→256) for better
    information preservation.

    Evidence: node3-1-1-1-1-2-1 feedback explicitly identifies the 10:1 compression
    as "likely creating an information bottleneck that negates ESM2-3B's advantage."
    """

    def __init__(self, in_dim: int = ESM2_DIM, mid_dim: int = 512, out_dim: int = STRING_DIM) -> None:
        super().__init__()
        self.proj1 = nn.Linear(in_dim, mid_dim, bias=True)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(mid_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.act(self.proj1(x)))


# ---------------------------------------------------------------------------
# Manifold Mixup Helper
# ---------------------------------------------------------------------------
def manifold_mixup(
    fused_emb: torch.Tensor,
    labels: Optional[torch.Tensor],
    alpha: float = 0.3,
    training: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
    """Apply Manifold Mixup in the fused embedding space."""
    if not training or labels is None:
        return fused_emb, None, 1.0

    B = fused_emb.shape[0]
    if B < 2:
        return fused_emb, None, 1.0

    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)

    perm = torch.randperm(B, device=fused_emb.device)
    mixed_emb = lam * fused_emb + (1.0 - lam) * fused_emb[perm]

    one_hot_a = F.one_hot(labels, num_classes=N_CLASSES).float()
    one_hot_b = F.one_hot(labels[perm], num_classes=N_CLASSES).float()
    mixed_labels = lam * one_hot_a + (1.0 - lam) * one_hot_b

    return mixed_emb, mixed_labels, lam


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def wce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Weighted cross-entropy with optional label smoothing.

    KEY CHANGE: Added label_smoothing parameter (default 0.05) to address
    the parent's 38.8× train/val loss ratio indicating over-confident logits.
    """
    return F.cross_entropy(
        logits.float(), labels,
        weight=class_weights.float(),
        label_smoothing=label_smoothing,
    )


def wce_loss_soft(
    logits: torch.Tensor,
    soft_labels: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Soft-label weighted cross-entropy for Mixup training (NO label smoothing needed)."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    ce = -(soft_labels.float() * log_probs).sum(dim=-1)
    if class_weights is not None:
        expected_cw = (soft_labels.float() * class_weights.float().unsqueeze(0)).sum(dim=-1)
        ce = ce * expected_cw
    return ce.mean()


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        # MLP head
        head_hidden_dim: int = HEAD_HIDDEN,
        head_inner_dim: int = HEAD_HIDDEN * 2,     # 768 inner dim
        head_dropout: float = 0.30,                # KEY CHANGE: 0.35 → 0.30 (tree-best level)
        n_blocks: int = 3,
        head_output_dropout: float = 0.10,
        # Label smoothing
        label_smoothing: float = 0.05,             # KEY CHANGE: new (0.05) to reduce overconfidence
        # Optimizers
        lr_muon: float = 0.01,
        lr_adamw_head: float = 3e-4,
        lr_string_near: float = 5e-5,
        lr_string_far: float = 3e-5,
        weight_decay: float = 2e-3,
        max_epochs: int = 600,                     # KEY CHANGE: 500 → 600 (third restart cycle)
        warmup_epochs: int = 10,
        n_string_nodes: int = 18870,
        # STRING fine-tuning: unfreeze last N GNN layers
        string_finetune_layers: int = 4,
        # Manifold Mixup
        mixup_alpha: float = 0.3,
        mixup_prob: float = 0.75,
        # CosineWarmRestarts
        t0_epochs: int = 80,
        t_mult: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone: Optional[nn.Module] = None
        self.string_gnn: Optional[nn.Module] = None
        self.cond_proj: Optional[nn.Module] = None  # TwoStepESM2Proj
        self.string_gain: Optional[nn.Parameter] = None
        self.gated_fusion: Optional[GatedFusion] = None
        self.input_proj: Optional[nn.Linear] = None
        self.blocks: Optional[nn.ModuleList] = None
        self.output_norm: Optional[nn.LayerNorm] = None
        self.head_dropout_layer: Optional[nn.Dropout] = None
        self.output_fc: Optional[nn.Linear] = None
        # No per_gene_bias (removed in parent lineage, kept removed)

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

        # For Muon parameter tracking
        self._muon_hidden_weights: List[nn.Parameter] = []
        self._adamw_non_hidden: List[nn.Parameter] = []

    def setup(self, stage: str = "fit") -> None:
        # ============================================================
        # 1. Load STRING_GNN with expanded fine-tuning (4 layers)
        # ============================================================
        self.print("Loading STRING_GNN model (expanded fine-tuning: last 4 layers + post_mp)...")
        string_gnn = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        string_gnn.eval()

        for param in string_gnn.parameters():
            param.requires_grad = False

        n_finetune = self.hparams.string_finetune_layers  # = 4
        total_layers = 8

        self._string_near_params: List[nn.Parameter] = []
        self._string_far_params: List[nn.Parameter] = []
        self._string_postmp_params: List[nn.Parameter] = []

        n_near = 2
        n_far = n_finetune - n_near

        def _matches_layer(param_name: str, layer_idx: int) -> bool:
            prefix = f"mps.{layer_idx}."
            return prefix in param_name or param_name.endswith(f"mps.{layer_idx}")

        unfrozen_params = 0
        for name, param in string_gnn.named_parameters():
            is_near = any(_matches_layer(name, idx) for idx in range(total_layers - n_near, total_layers))
            is_far = any(_matches_layer(name, idx) for idx in range(total_layers - n_finetune, total_layers - n_near))
            is_postmp = "post_mp" in name

            if is_near:
                param.requires_grad = True
                self._string_near_params.append(param)
                unfrozen_params += param.numel()
            elif is_far:
                param.requires_grad = True
                self._string_far_params.append(param)
                unfrozen_params += param.numel()
            elif is_postmp:
                param.requires_grad = True
                self._string_postmp_params.append(param)
                unfrozen_params += param.numel()

        for name, param in string_gnn.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        total_string = sum(p.numel() for p in string_gnn.parameters())
        self.print(
            f"STRING_GNN: {total_string:,} total params, "
            f"{unfrozen_params:,} trainable (last {n_finetune} GCN layers + post_mp)"
        )
        self.string_gnn = string_gnn

        graph_data = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
        edge_index = graph_data["edge_index"]
        edge_weight = graph_data.get("edge_weight", None)
        self.register_buffer("edge_index", edge_index.long())
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight.float())
        else:
            self.register_buffer("edge_weight", None)

        # ============================================================
        # 2. Load FROZEN ESM2-3B
        # ============================================================
        self.print("Loading FROZEN ESM2-3B (2560-dim, no fine-tuning)...")
        backbone = EsmForMaskedLM.from_pretrained(ESM2_MODEL, dtype=torch.bfloat16)

        for param in backbone.parameters():
            param.requires_grad = False

        backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "config"):
            backbone.config.use_cache = False

        self.backbone = backbone
        total_esm2 = sum(p.numel() for p in self.backbone.parameters())
        self.print(f"ESM2-3B: {total_esm2:,} total params, all FROZEN")

        # ============================================================
        # 3. Two-step ESM2 projection + gated fusion + PreNorm MLP
        #    KEY CHANGE: cond_proj is TwoStepESM2Proj (2560→512→256) [inherited from parent]
        #    KEY CHANGE: head_dropout reduced 0.35 → 0.30 (tree-best level)
        # ============================================================
        # TwoStepESM2Proj: reduces 10:1 compression bottleneck to two 5:1 steps
        self.cond_proj = TwoStepESM2Proj(in_dim=ESM2_DIM, mid_dim=512, out_dim=STRING_DIM)

        self.string_gain = nn.Parameter(torch.ones(1))

        self.gated_fusion = GatedFusion(
            protein_dim=ESM2_DIM,
            string_dim=STRING_DIM,
            fusion_dim=FUSION_DIM,
        )

        self.input_proj = nn.Linear(FUSION_DIM, self.hparams.head_hidden_dim, bias=True)

        # KEY CHANGE: head_dropout reduced from 0.35 → 0.30 (tree-best level)
        self.blocks = nn.ModuleList([
            PreNormResBlock(
                hidden_dim=self.hparams.head_hidden_dim,
                inner_dim=self.hparams.head_inner_dim,
                dropout=self.hparams.head_dropout,
            )
            for _ in range(self.hparams.n_blocks)
        ])

        self.output_norm = nn.LayerNorm(self.hparams.head_hidden_dim)
        self.head_dropout_layer = nn.Dropout(self.hparams.head_output_dropout)
        self.output_fc = nn.Linear(self.hparams.head_hidden_dim, N_GENES * N_CLASSES, bias=True)
        # No per_gene_bias (kept removed from parent lineage)

        # Cast trainable params to float32
        for mod in [self.cond_proj, self.gated_fusion, self.input_proj, self.blocks, self.output_norm, self.output_fc]:
            if mod is not None:
                for p in mod.parameters():
                    p.data = p.data.float()
        self.string_gain.data = self.string_gain.data.float()

        # ============================================================
        # 4. Muon parameter classification
        # ============================================================
        self._muon_hidden_weights = []
        self._adamw_non_hidden = []

        for block in self.blocks:
            for name, p in block.named_parameters():
                if p.requires_grad:
                    if p.ndim >= 2 and ("fc1.weight" in name or "fc2.weight" in name):
                        self._muon_hidden_weights.append(p)
                    else:
                        self._adamw_non_hidden.append(p)

        if self.input_proj.weight.ndim >= 2:
            self._muon_hidden_weights.append(self.input_proj.weight)
        if self.input_proj.bias is not None:
            self._adamw_non_hidden.append(self.input_proj.bias)

        for p in self.output_fc.parameters():
            self._adamw_non_hidden.append(p)

        for mod in [self.cond_proj, self.gated_fusion, self.output_norm]:
            for p in mod.parameters():
                if p.requires_grad:
                    self._adamw_non_hidden.append(p)
        self._adamw_non_hidden.append(self.string_gain)

        # ============================================================
        # 5. Loss: Weighted Cross-Entropy with optional label smoothing
        #    Correct class weights: class 0=down(4.77%), class 1=neutral(92.82%), class 2=up(2.41%)
        #    KEY CHANGE: label_smoothing=0.05 added to address 38.8× train/val loss ratio
        # ============================================================
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        weights = 1.0 / freq
        weights = weights / weights.mean()
        self.register_buffer("class_weights", weights)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"FrozenESM2-3B + "
            f"STRING_GNN(4-layer ft) + "
            f"TwoStepProj(2560→512→256) + "
            f"GatedFusion(dim={FUSION_DIM}) + "
            f"PreNorm-MLP(h={self.hparams.head_hidden_dim}x{self.hparams.n_blocks}) + "
            f"dropout={self.hparams.head_dropout} + "
            f"label_smoothing={self.hparams.label_smoothing} + "
            f"wd={self.hparams.weight_decay} | "
            f"trainable={trainable:,}/{total:,} params "
            f"({100.0 * trainable / total:.2f}%)"
        )

    def _get_string_embeddings_with_cond(
        self,
        protein_emb: torch.Tensor,
        string_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Run STRING_GNN with protein-conditioned cond_emb.

        Uses TwoStepESM2Proj(2560→512→256) instead of Linear(2560→256).
        """
        n_nodes = self.hparams.n_string_nodes
        device = protein_emb.device

        # Two-step projection: 2560 → 512 → 256 (reduces compression bottleneck)
        cond_per_sample = self.cond_proj(protein_emb.float())  # [B, 256]
        cond_per_sample = cond_per_sample * self.string_gain   # learnable gain

        valid_mask = string_idx < n_nodes
        cond_emb = torch.zeros(n_nodes, STRING_DIM, device=device, dtype=cond_per_sample.dtype)
        if valid_mask.any():
            valid_idx = string_idx[valid_mask]
            valid_cond = cond_per_sample[valid_mask]
            cond_emb = cond_emb.index_add(0, valid_idx, valid_cond)

        outputs = self.string_gnn(
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            cond_emb=cond_emb,
        )
        string_embs = outputs.last_hidden_state  # [n_nodes, 256]

        pad = torch.zeros(1, STRING_DIM, dtype=string_embs.dtype, device=device)
        string_embs_padded = torch.cat([string_embs, pad], dim=0)
        return string_embs_padded[string_idx]  # [B, 256]

    def _encode(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Encode a batch → [B, FUSION_DIM] via gated fusion."""
        with torch.no_grad():
            out = self.backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
        hidden = out.hidden_states[-1].float()  # [B, T, 2560]

        attn_mask = batch["attention_mask"].float()
        seq_lens = attn_mask.sum(dim=1).long()
        B, T = attn_mask.shape
        positions = torch.arange(T, device=attn_mask.device).unsqueeze(0).expand(B, -1)
        eos_pos = (seq_lens - 1).unsqueeze(1)
        special_mask = (positions == 0) | (positions == eos_pos)
        valid_mask = attn_mask.bool() & ~special_mask

        valid_float = valid_mask.float()
        protein_emb = (hidden * valid_float.unsqueeze(-1)).sum(dim=1)
        count = valid_float.sum(dim=1, keepdim=True).clamp(min=1e-9)
        protein_emb = protein_emb / count  # [B, 2560]

        string_emb = self._get_string_embeddings_with_cond(protein_emb, batch["string_idx"])
        return self.gated_fusion(protein_emb, string_emb)

    def _forward_head(self, fused_emb: torch.Tensor) -> torch.Tensor:
        """Apply 3-block PreNorm MLP + flat head → [B, N_CLASSES, N_GENES]."""
        x = self.input_proj(fused_emb)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        x = self.head_dropout_layer(x)
        out = self.output_fc(x)
        out = out.view(-1, N_CLASSES, N_GENES)
        return out

    def _compute_loss_hard(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Weighted CE with label smoothing for hard labels.

        KEY CHANGE: label_smoothing applied here to reduce over-confident logits.
        Parent had 38.8× train/val loss ratio; smoothing addresses this directly.
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return wce_loss(
            logits_flat, labels_flat, self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    def _compute_loss_soft(self, logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
        """Soft-label WCE for Mixup (no additional label smoothing — already soft)."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        soft_flat = soft_labels.reshape(-1, N_CLASSES).float()
        return wce_loss_soft(logits_flat, soft_flat, self.class_weights)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        fused_emb = self._encode(batch)

        apply_mixup = (
            self.hparams.mixup_prob > 0
            and torch.rand(1).item() < self.hparams.mixup_prob
            and "label" in batch
            and batch["label"].shape[0] >= 2
        )

        if apply_mixup:
            mixed_emb, mixed_labels, lam = manifold_mixup(
                fused_emb,
                batch["label"],
                alpha=self.hparams.mixup_alpha,
                training=True,
            )
            logits = self._forward_head(mixed_emb)
            if mixed_labels is not None:
                loss = self._compute_loss_soft(logits, mixed_labels)
            else:
                loss = self._compute_loss_hard(logits, batch["label"])
        else:
            logits = self._forward_head(fused_emb)
            loss = self._compute_loss_hard(logits, batch["label"])

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        fused_emb = self._encode(batch)
        logits = self._forward_head(fused_emb)
        # Validation loss: use label_smoothing=0 for clean loss tracking
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = batch["label"].reshape(-1)
        loss = wce_loss(logits_flat, labels_flat, self.class_weights, label_smoothing=0.0)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds_local = torch.cat(self._val_preds, dim=0)
        labels_local = torch.cat(self._val_labels, dim=0)
        self._val_preds.clear()
        self._val_labels.clear()

        all_preds = self.all_gather(preds_local)
        all_labels = self.all_gather(labels_local)
        if self.trainer.world_size > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
            all_labels = all_labels.view(-1, N_GENES)
        else:
            all_preds = all_preds.squeeze(0)
            all_labels = all_labels.squeeze(0)

        f1 = _compute_per_gene_f1(
            all_preds.float().cpu().numpy(), all_labels.cpu().numpy()
        )
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        fused_emb = self._encode(batch)
        logits = self._forward_head(fused_emb)
        self._test_preds.append(logits.detach().cpu().float())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist

        preds_local = torch.cat(self._test_preds, dim=0)
        self._test_preds.clear()

        all_preds = self.all_gather(preds_local)
        world_size = self.trainer.world_size
        if world_size > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
        else:
            all_preds = all_preds.squeeze(0)
        local_ids = list(self._test_pert_ids)
        local_syms = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        all_pert_ids: List[str] = []
        all_symbols: List[str] = []

        if world_size > 1:
            local_ids_bytes = pickle.dumps(local_ids)
            local_syms_bytes = pickle.dumps(local_syms)
            local_ids_len = len(local_ids_bytes)
            local_syms_len = len(local_syms_bytes)

            ids_lens = [0] * world_size
            syms_lens = [0] * world_size
            dist.all_gather_object(ids_lens, local_ids_len)
            dist.all_gather_object(syms_lens, local_syms_len)

            max_ids_len = max(ids_lens) if ids_lens else 0
            max_syms_len = max(syms_lens) if syms_lens else 0

            if max_ids_len > 0:
                ids_np = np.frombuffer(local_ids_bytes, dtype=np.uint8).copy()
                ids_tensor = F.pad(torch.from_numpy(ids_np).cuda(), (0, max_ids_len - local_ids_len))
                gathered_ids = [torch.zeros(max_ids_len, dtype=torch.uint8, device="cuda") for _ in range(world_size)]
                dist.all_gather(gathered_ids, ids_tensor)
                for r, blen in enumerate(ids_lens):
                    if blen > 0:
                        all_pert_ids.extend(pickle.loads(gathered_ids[r][:blen].cpu().numpy().tobytes()))

            if max_syms_len > 0:
                syms_np = np.frombuffer(local_syms_bytes, dtype=np.uint8).copy()
                syms_tensor = F.pad(torch.from_numpy(syms_np).cuda(), (0, max_syms_len - local_syms_len))
                gathered_syms = [torch.zeros(max_syms_len, dtype=torch.uint8, device="cuda") for _ in range(world_size)]
                dist.all_gather(gathered_syms, syms_tensor)
                for r, blen in enumerate(syms_lens):
                    if blen > 0:
                        all_symbols.extend(pickle.loads(gathered_syms[r][:blen].cpu().numpy().tobytes()))
        else:
            all_pert_ids = local_ids
            all_symbols = local_syms

        has_labels = bool(self._test_labels)
        all_labels = None
        if has_labels:
            labels_local = torch.cat(self._test_labels, dim=0)
            self._test_labels.clear()
            all_labels_gathered = self.all_gather(labels_local)
            if world_size > 1:
                all_labels = all_labels_gathered.view(-1, N_GENES)
            else:
                all_labels = all_labels_gathered.squeeze(0)

        # De-duplicate on ALL ranks (all_gather gives each rank the full gathered data)
        n_preds = all_preds.shape[0]
        n_ids = len(all_pert_ids)
        min_len = min(n_preds, n_ids)

        seen: set = set()
        unique_idx: List[int] = []
        for i, pid in enumerate(all_pert_ids[:min_len]):
            if pid not in seen:
                seen.add(pid)
                unique_idx.append(i)

        dedup_preds = all_preds[unique_idx]
        dedup_ids = [all_pert_ids[i] for i in unique_idx]
        dedup_syms = [all_symbols[i] for i in unique_idx]

        # Only rank 0 writes the prediction file
        if self.trainer.is_global_zero:
            _save_test_predictions(
                pert_ids=dedup_ids,
                symbols=dedup_syms,
                preds=dedup_preds.float().cpu().numpy(),
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

        # Compute F1 on all ranks (same gathered data); use sync_dist=True so Lightning
        # does not warn about missing cross-rank synchronization. All ranks have the
        # same value after all_gather, so averaging over ranks is a no-op numerically.
        if has_labels and all_labels is not None:
            dedup_labels = all_labels[unique_idx]
            f1 = _compute_per_gene_f1(
                dedup_preds.float().cpu().numpy(),
                dedup_labels.cpu().numpy(),
            )
            self.log("test_f1", f1, prog_bar=True, sync_dist=True)
            self.print(f"Test F1: {f1:.4f}")

    def configure_optimizers(self):
        """Muon+AdamW optimizer with CosineWarmRestarts + warmup.

        KEY CHANGES vs parent:
        - max_epochs=600 (was 500): extends training to capture third warm restart cycle
        - T_0=80, T_mult=2: restarts at epochs 80, 240 within 600-epoch budget
        - All other optimizer settings unchanged
        """
        from muon import MuonWithAuxAdam

        string_near_params = self._string_near_params + self._string_postmp_params
        string_far_params = self._string_far_params

        param_groups = [
            dict(
                params=self._muon_hidden_weights,
                use_muon=True,
                lr=self.hparams.lr_muon,
                weight_decay=self.hparams.weight_decay,
                momentum=0.95,
            ),
            dict(
                params=self._adamw_non_hidden,
                use_muon=False,
                lr=self.hparams.lr_adamw_head,
                betas=(0.9, 0.95),
                weight_decay=self.hparams.weight_decay,
            ),
        ]

        if string_near_params:
            param_groups.append(dict(
                params=string_near_params,
                use_muon=False,
                lr=self.hparams.lr_string_near,
                betas=(0.9, 0.95),
                weight_decay=self.hparams.weight_decay,
            ))

        if string_far_params:
            param_groups.append(dict(
                params=string_far_params,
                use_muon=False,
                lr=self.hparams.lr_string_far,
                betas=(0.9, 0.95),
                weight_decay=self.hparams.weight_decay,
            ))

        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        optimizer = MuonWithAuxAdam(param_groups)

        # CosineWarmRestarts + linear warmup
        # T_0=80, T_mult=2 → restarts at epochs 80, 240 within 600-epoch budget
        # Third cycle: epochs 240–560; extended from parent's 500 to fully utilize it
        T_0 = self.hparams.t0_epochs        # = 80
        T_mult = self.hparams.t_mult        # = 2
        warmup_epochs = self.hparams.warmup_epochs  # = 10

        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < warmup_epochs:
                return float(current_epoch) / float(max(1, warmup_epochs))

            epoch = current_epoch - warmup_epochs
            t_cur = epoch
            t_i = T_0
            while t_cur >= t_i:
                t_cur -= t_i
                t_i = int(t_i * T_mult)

            return max(5e-3, 0.5 * (1.0 + np.cos(np.pi * t_cur / t_i)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save trainable parameters and persistent buffers."""
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys = {prefix + n for n, _ in self.named_buffers()}
        save_keys = trainable_keys | buffer_keys
        result = {k: v for k, v in full_sd.items() if k in save_keys}

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buf_total = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable}/{total} params "
            f"({100.0 * trainable / total:.2f}%), plus {buf_total} buffer values"
        )
        return result

    def load_state_dict(self, state_dict, strict=True):
        """Load partial checkpoint (trainable params + buffers only)."""
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro F1, averaged over all n_genes (matches calc_metric.py)."""
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [n_samples, n_genes]
    n_genes = labels.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = labels[:, g].flatten().astype(np.int32)
        yh = y_hat[:, g].flatten().astype(np.int32)
        per_class = sk_f1(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        f1_vals.append(float(per_class[present].mean()) if present.any() else 0.0)
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    n = min(len(pert_ids), len(preds))
    for i in range(n):
        p = preds[i]
        if p.ndim == 3:
            p = p.squeeze(0)
        elif p.ndim == 4:
            p = p.squeeze(0).squeeze(0)
        rows.append({
            "idx": pert_ids[i],
            "input": symbols[i],
            "prediction": json.dumps(p.tolist()),
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}")


def _ensemble_checkpoints_and_test(
    model: PerturbModule,
    datamodule: PerturbDataModule,
    checkpoint_dir: Path,
    output_dir: Path,
    top_k: int = 9,
) -> None:
    """Load top-K checkpoints, average their logit predictions, and save.

    KEY CHANGE: top_k increased from 7 → 9 for larger ensemble diversity.
    With max_epochs=600 and three full warm restart cycles, more high-quality
    checkpoints are available spanning cycle 2 and cycle 3 peaks.
    """
    ckpt_files = sorted(
        [f for f in checkpoint_dir.rglob("*.ckpt") if "last" not in f.name],
        key=lambda x: x.stat().st_mtime,
        reverse=False,
    )

    def extract_f1(f: Path) -> float:
        name = f.stem
        for sep in ["f1=", "val_f1=", "val_f1_"]:
            if sep in name:
                try:
                    f1_str = name.split(sep)[-1].split("-")[0].split("_")[0].split(".ckpt")[0]
                    return float(f1_str)
                except (IndexError, ValueError):
                    pass
        return 0.0

    ckpt_files = sorted(ckpt_files, key=extract_f1, reverse=True)
    top_k_actual = min(top_k, len(ckpt_files))

    if top_k_actual == 0:
        print("No checkpoints found for ensembling, skipping.")
        return

    print(f"Checkpoint ensemble: using top-{top_k_actual} checkpoints by val_f1")
    for i, f in enumerate(ckpt_files[:top_k_actual]):
        print(f"  [{i+1}] {f.name} (val_f1={extract_f1(f):.4f})")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_ensemble_preds: List[np.ndarray] = []
    all_pert_ids: Optional[List[str]] = None
    all_symbols: Optional[List[str]] = None

    for ckpt_path in ckpt_files[:top_k_actual]:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()

        preds_list = []
        pert_ids_list = []
        symbols_list = []

        test_loader = datamodule.test_dataloader()

        with torch.no_grad():
            for batch in test_loader:
                batch_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                fused_emb = model._encode(batch_device)
                logits = model._forward_head(fused_emb)
                preds_list.append(logits.cpu().float().numpy())
                pert_ids_list.extend(batch["pert_id"])
                symbols_list.extend(batch["symbol"])

        preds_arr = np.concatenate(preds_list, axis=0)
        all_ensemble_preds.append(preds_arr)

        if all_pert_ids is None:
            all_pert_ids = pert_ids_list
            all_symbols = symbols_list

    avg_preds = np.mean(all_ensemble_preds, axis=0)

    if all_pert_ids is not None:
        seen = set()
        unique_idx = []
        for i, pid in enumerate(all_pert_ids):
            if pid not in seen:
                seen.add(pid)
                unique_idx.append(i)

        _save_test_predictions(
            pert_ids=[all_pert_ids[i] for i in unique_idx],
            symbols=[all_symbols[i] for i in unique_idx],
            preds=avg_preds[unique_idx],
            out_path=output_dir / "test_predictions.tsv",
        )
        print(f"Checkpoint ensemble predictions saved ({top_k_actual} checkpoints averaged).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Node2-1-2-1-1-1-1-1-1-1: Frozen ESM2-3B + STRING_GNN (4-layer ft) + "
            "TwoStepProj(2560→512→256) + Muon + GatedFusion(512) + PreNorm MLP "
            "(h=384, dropout=0.30) + WD=2e-3 + label_smooth=0.05 + no per_gene_bias + "
            "T_0=80 + mixup=0.75 + max_epochs=600 + patience=130"
        )
    )
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=600)            # KEY: 500 → 600
    p.add_argument("--lr-muon", type=float, default=0.01)
    p.add_argument("--lr-adamw-head", type=float, default=3e-4)
    p.add_argument("--lr-string-near", type=float, default=5e-5)
    p.add_argument("--lr-string-far", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=2e-3)
    p.add_argument("--head-hidden-dim", type=int, default=HEAD_HIDDEN)
    p.add_argument("--head-inner-dim", type=int, default=HEAD_HIDDEN * 2)
    p.add_argument("--head-dropout", type=float, default=0.30)       # KEY: 0.35 → 0.30
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--head-output-dropout", type=float, default=0.10)
    p.add_argument("--label-smoothing", type=float, default=0.05)    # KEY: NEW (0.05)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--early-stop-patience", type=int, default=130)   # KEY: 100 → 130
    p.add_argument("--string-finetune-layers", type=int, default=4)
    p.add_argument("--mixup-alpha", type=float, default=0.3)
    p.add_argument("--mixup-prob", type=float, default=0.75)
    p.add_argument("--t0-epochs", type=int, default=80)
    p.add_argument("--t-mult", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--no-ensemble", action="store_true", help="Skip checkpoint ensembling at test time")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        max_seq_len=MAX_SEQ_LEN,
    )

    _node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    n_string_nodes = len(_node_names)

    model = PerturbModule(
        head_hidden_dim=args.head_hidden_dim,
        head_inner_dim=args.head_inner_dim,
        head_dropout=args.head_dropout,
        n_blocks=args.n_blocks,
        head_output_dropout=args.head_output_dropout,
        label_smoothing=args.label_smoothing,
        lr_muon=args.lr_muon,
        lr_adamw_head=args.lr_adamw_head,
        lr_string_near=args.lr_string_near,
        lr_string_far=args.lr_string_far,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
        n_string_nodes=n_string_nodes,
        string_finetune_layers=args.string_finetune_layers,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        t0_epochs=args.t0_epochs,
        t_mult=args.t_mult,
    )

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = limit_val = limit_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-ep={epoch:03d}-f1={val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=9,   # KEY CHANGE: 7 → 9 for larger ensemble with 3 warm restart cycles
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stop_patience,   # KEY CHANGE: 130 (was 100 in parent)
        min_delta=1e-5,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=180)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=args.val_check_interval if (args.debug_max_step is None and not args.fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        gradient_clip_val=1.0,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        best_ckpt = checkpoint_cb.best_model_path
        if not best_ckpt or not Path(best_ckpt).exists():
            best_ckpt = str(output_dir / "checkpoints" / "last.ckpt")
        print(f"Test checkpoint path (best): {best_ckpt}")
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path=best_ckpt)

        # Checkpoint ensemble (top-9 for larger diverse ensemble)
        use_ensemble = not args.no_ensemble
        checkpoint_dir = output_dir / "checkpoints"

        if use_ensemble and checkpoint_dir.exists() and trainer.is_global_zero:
            print("\nRunning checkpoint ensemble for improved test predictions...")
            _ensemble_checkpoints_and_test(
                model=model,
                datamodule=datamodule,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                top_k=9,    # KEY CHANGE: 7 → 9
            )

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        result_dict = test_results[0] if isinstance(test_results, list) else test_results
        if isinstance(result_dict, dict):
            score_value = result_dict.get(
                "test_f1",
                result_dict.get("test/f1",
                result_dict.get("test/metric",
                result_dict.get("f1",
                result_dict.get("metric")))))
        else:
            score_value = float(result_dict)
        score_path.write_text(str(score_value))
        print(f"Test score → {score_path}: {score_value}")


if __name__ == "__main__":
    main()
