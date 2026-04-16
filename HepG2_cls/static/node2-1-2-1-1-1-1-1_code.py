"""Node 2-1-2-1-1-1-1-1: Aggressive regularization + ESM2-3B upgrade
         from parent node2-1-2-1-1-1-1 (Test F1=0.5196).

Key improvements over parent node2-1-2-1-1-1-1 (Test F1=0.5196):

  1. UPGRADE ESM2-650M → ESM2-3B (HIGHEST PRIORITY)
     - Tree-best (0.5283) uses ESM2-3B; parent used ESM2-650M reaching only 0.5196
     - ESM2-3B produces 2560-dim embeddings (vs 1280-dim for 650M)
     - Richer protein representations → better separation of gene perturbation effects
     - GatedFusion protein_dim updated: 1280 → 2560
     - Memory: ESM2-3B in bf16 = ~5.4GB, leaves ample VRAM for training on H100

  2. INCREASE REGULARIZATION: head_dropout 0.30→0.45, weight_decay 8e-4→2e-3 (HIGHEST PRIORITY)
     - Parent showed severe 8.5× train-val loss divergence (train=0.030, val=0.259)
     - Parent feedback recommendation #1: "Increase head_dropout from 0.30 → 0.45"
     - Parent feedback recommendation #2: "Increase weight_decay from 8e-4 → 1.5e-3 or 2e-3"
     - This directly attacks the overfitting bottleneck preventing generalization beyond 0.52

  3. INCREASE head_output_dropout: 0.10 → 0.15 (HIGH PRIORITY)
     - Parent feedback recommendation: add head_output_dropout=0.15
     - Final layer dropout was too light given the 8.5× overfitting gap

  4. REMOVE per_gene_bias (HIGH PRIORITY)
     - per_gene_bias is 19,920 overfitting-prone parameters (3 × 6640)
     - Parent feedback: "Consider reducing focal gamma to 0.5 or removing per_gene_bias"
     - With only 1,273 training samples, per-gene bias directly memorizes training statistics
     - The output_fc Linear(384 → 19920) already models per-gene capacity without bias

  5. INCREASE Mixup alpha: 0.2 → 0.3 (MEDIUM PRIORITY)
     - Parent feedback: "Mixup alpha increase (0.2 → 0.3) with stronger dropout"
     - Slightly stronger interpolation compensates for higher dropout regularization
     - Provides smoother embedding manifold with less risk of mode collapse

  6. REDUCE early stopping patience: 80 → 50 (MEDIUM PRIORITY)
     - Parent feedback: "Reduce early stopping patience from 80 → 40-50"
     - Parent's best val_f1 was at epoch 160; training continued 79 more epochs wastefully
     - Shorter patience prevents wasting compute on post-peak overfitting

  7. Keep all proven architectural components from parent (UNCHANGED):
     - Frozen ESM2 (now 3B, not 650M, still fully frozen)
     - STRING_GNN 4-layer fine-tuning with tiered LR (near=5e-5, far=3e-5)
     - GatedFusion(FUSION_DIM=512)
     - 3-block PreNorm MLP (h=384, inner=768)
     - MuonWithAuxAdam optimizer
     - LambdaLR CosineWarmRestarts (T_0=100, T_mult=2)
     - Manifold Mixup (prob=0.65)
     - WCE loss with inverse-frequency class weights
     - Top-5 checkpoint ensemble

Memory rationale:
  - node2-1-2-1-1-1-1 (parent, F1=0.5196): 8.5x overfitting, head_dropout too low
  - node3-1-1-1-1-2-1-1-1 (tree-best, F1=0.5283): ESM2-3B + frozen STRING + Muon
  - node3-3-1-2-1-1-1 (F1=0.5243): ESM2-650M frozen → proven ceiling with 650M
  - parent feedback.md: Priority 1 = increase regularization; Priority 2 = ESM2-3B

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
# KEY CHANGE: Upgrade from ESM2-650M (1280-dim) to ESM2-3B (2560-dim)
# Tree-best node3-1-1-1-1-2-1-1-1 (F1=0.5283) uses ESM2-3B
ESM2_MODEL = "facebook/esm2_t36_3B_UR50D"
PROTEIN_FASTA = "/home/data/genome/hg38_gencode_protein.fa"
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")

N_GENES = 6640
N_CLASSES = 3
MAX_SEQ_LEN = 512       # ESM2 max context, enough for most human proteins
ESM2_DIM = 2560         # ESM2-3B hidden size (vs 1280 for 650M)
STRING_DIM = 256        # STRING_GNN output dimension
FUSION_DIM = 512        # Fusion dimension (wider than parent's 256; proven in tree-best)
HEAD_HIDDEN = 384       # PreNorm MLP hidden dimension (tree-best proven capacity)
FALLBACK_SEQ = "M"      # Minimal placeholder if ENSG not in FASTA
STRING_FALLBACK_IDX = 18870  # Zero-row index for genes absent from STRING


# ---------------------------------------------------------------------------
# Protein sequence lookup helpers
# ---------------------------------------------------------------------------
def _build_ensg_to_seq(fasta_path: str) -> Dict[str, str]:
    """Parse hg38_gencode_protein.fa → ENSG (no version) → longest protein sequence.

    Header format: >ENSP...|ENST...|ENSG00000186092.7|...
    Field index 2 (after '|') contains ENSG with version suffix.
    """
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

        # STRING_GNN node indices (STRING_FALLBACK_IDX for missing genes)
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
        # --- Tokenizer: rank-0 downloads first, all ranks load ---
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(ESM2_MODEL, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL, trust_remote_code=True)

        # --- Protein FASTA ---
        self.ensg2seq = get_ensg2seq()

        # --- STRING node names: ENSG_ID → STRING node index ---
        node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        self.n_string_nodes = len(node_names)
        self.ensg_to_string_idx = {name: i for i, name in enumerate(node_names)}

        # --- Datasets ---
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

    KEY CHANGE: protein_dim updated to 2560 for ESM2-3B (was 1280 for ESM2-650M).
    Both branches are projected to FUSION_DIM=512.
    A gate vector learned from the concatenation selects between the two modalities.
    Output: [B, FUSION_DIM] fused representation.

    FUSION_DIM=512 proven superior to 256 in tree-best node3-1-1-1-1-2-1-1 (+0.002 F1).
    """

    def __init__(
        self,
        protein_dim: int = ESM2_DIM,  # 2560 for ESM2-3B
        string_dim: int = STRING_DIM,
        fusion_dim: int = FUSION_DIM,
    ) -> None:
        super().__init__()
        self.protein_proj = nn.Linear(protein_dim, fusion_dim, bias=True)
        self.protein_norm = nn.LayerNorm(fusion_dim)
        self.string_proj = nn.Linear(string_dim, fusion_dim, bias=True)
        self.string_norm = nn.LayerNorm(fusion_dim)
        # Learnable gate combining both modalities
        self.gate_fc = nn.Linear(fusion_dim * 2, fusion_dim, bias=True)
        nn.init.zeros_(self.gate_fc.bias)

    def forward(self, protein_emb: torch.Tensor, string_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            protein_emb: [B, ESM2_DIM] from mean pooling (2560 for ESM2-3B)
            string_emb: [B, STRING_DIM] from STRING_GNN lookup
        Returns:
            fused: [B, FUSION_DIM]
        """
        p = self.protein_norm(self.protein_proj(protein_emb.float()))   # [B, FUSION_DIM]
        s = self.string_norm(self.string_proj(string_emb.float()))       # [B, FUSION_DIM]
        concat = torch.cat([p, s], dim=-1)                               # [B, 2*FUSION_DIM]
        gate = torch.sigmoid(self.gate_fc(concat))                       # [B, FUSION_DIM]
        return gate * p + (1.0 - gate) * s                               # [B, FUSION_DIM]


# ---------------------------------------------------------------------------
# PreNorm Residual Block (tree-best architecture)
# ---------------------------------------------------------------------------
class PreNormResBlock(nn.Module):
    """Pre-normalization residual block for MLP.

    Structure: LN → Linear → GELU → Dropout → Linear → + residual

    Proven in tree-best nodes (F1=0.5243) as the optimal capacity unit
    for STRING+ESM2 fusion on 1,273 samples.
    h=384 is the proven sweet spot (h=512/4-blocks causes regression).
    """

    def __init__(self, hidden_dim: int, inner_dim: int, dropout: float = 0.45) -> None:
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
# Manifold Mixup Helper
# ---------------------------------------------------------------------------
def manifold_mixup(
    fused_emb: torch.Tensor,
    labels: Optional[torch.Tensor],
    alpha: float = 0.3,
    training: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
    """Apply Manifold Mixup in the fused embedding space.

    Interpolates between pairs of samples in the FUSION_DIM-dim fused representation.
    This provides strong regularization without corrupting the structured
    protein + PPI signal that the encoder produces.

    KEY CHANGE: alpha increased from 0.2 → 0.3 (stronger interpolation per parent feedback)

    Args:
        fused_emb: [B, FUSION_DIM] fused embeddings
        labels: [B, N_GENES] integer class labels in {0, 1, 2}, or None for test
        alpha: Dirichlet concentration parameter (0.3 gives slightly more mixing)
        training: whether to apply Mixup (disabled at eval/test time)

    Returns:
        mixed_emb: [B, FUSION_DIM] mixed embeddings
        mixed_labels: [B, N_GENES, N_CLASSES] soft one-hot labels, or None
        lam: mixing coefficient
    """
    if not training or labels is None:
        return fused_emb, None, 1.0

    B = fused_emb.shape[0]
    if B < 2:
        return fused_emb, None, 1.0

    # Sample mixing coefficient from Beta distribution
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5 for stability

    # Random permutation for the second sample
    perm = torch.randperm(B, device=fused_emb.device)

    # Mix embeddings
    mixed_emb = lam * fused_emb + (1.0 - lam) * fused_emb[perm]

    # Mix labels: convert integer labels to one-hot, then interpolate
    one_hot_a = F.one_hot(labels, num_classes=N_CLASSES).float()         # [B, N_GENES, 3]
    one_hot_b = F.one_hot(labels[perm], num_classes=N_CLASSES).float()   # [B, N_GENES, 3]
    mixed_labels = lam * one_hot_a + (1.0 - lam) * one_hot_b             # [B, N_GENES, 3]

    return mixed_emb, mixed_labels, lam


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def wce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted cross-entropy loss with class weights.

    Args:
        logits: [B*N_GENES, N_CLASSES] float32 logits
        labels: [B*N_GENES] long labels in {0, 1, 2}
        class_weights: [N_CLASSES] float32 per-class weights
    """
    return F.cross_entropy(logits.float(), labels, weight=class_weights.float())


def wce_loss_soft(
    logits: torch.Tensor,
    soft_labels: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted cross-entropy loss with soft (Mixup-interpolated) labels.

    Used for Manifold Mixup where labels are interpolated between two samples.

    Args:
        logits: [B*N_GENES, N_CLASSES] float32 logits
        soft_labels: [B*N_GENES, N_CLASSES] float32 soft class probabilities
        class_weights: [N_CLASSES] float32 per-class weights
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)   # [B, C]

    # Soft cross-entropy
    ce = -(soft_labels.float() * log_probs).sum(dim=-1)  # [B]

    # Apply class weights: weighted by expected class (expectation over soft distribution)
    if class_weights is not None:
        expected_cw = (soft_labels.float() * class_weights.float().unsqueeze(0)).sum(dim=-1)  # [B]
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
        head_inner_dim: int = HEAD_HIDDEN * 2,  # 768 inner dim
        head_dropout: float = 0.45,             # KEY CHANGE: 0.30 → 0.45 (parent feedback)
        n_blocks: int = 3,
        head_output_dropout: float = 0.15,      # KEY CHANGE: 0.10 → 0.15 (parent feedback)
        # Optimizers
        lr_muon: float = 0.01,          # Muon LR for hidden MLP weights
        lr_adamw_head: float = 3e-4,    # AdamW LR for head output + cond_proj etc.
        lr_string_near: float = 5e-5,   # STRING near layers (mps.6,7 + post_mp)
        lr_string_far: float = 3e-5,    # STRING far layers (mps.4,5)
        weight_decay: float = 2e-3,     # KEY CHANGE: 8e-4 → 2e-3 (parent feedback)
        max_epochs: int = 400,
        warmup_epochs: int = 10,
        n_string_nodes: int = 18870,
        # STRING fine-tuning: unfreeze last N GNN layers (4 from parent)
        string_finetune_layers: int = 4,
        # Manifold Mixup
        mixup_alpha: float = 0.3,       # KEY CHANGE: 0.2 → 0.3 (parent feedback)
        mixup_prob: float = 0.65,       # Proven in tree-best (vs parent's 0.50)
        # CosineWarmRestarts
        t0_epochs: int = 100,           # First cycle: 100 epochs (longer than parent's 80)
        t_mult: int = 2,                # T_0=100, T_mult=2 → restarts at 100, 300
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone: Optional[nn.Module] = None
        self.string_gnn: Optional[nn.Module] = None
        self.cond_proj: Optional[nn.Linear] = None
        self.string_gain: Optional[nn.Parameter] = None
        self.gated_fusion: Optional[GatedFusion] = None
        self.input_proj: Optional[nn.Linear] = None
        self.blocks: Optional[nn.ModuleList] = None
        self.output_norm: Optional[nn.LayerNorm] = None
        self.head_dropout_layer: Optional[nn.Dropout] = None
        self.output_fc: Optional[nn.Linear] = None
        # NOTE: per_gene_bias REMOVED to reduce overfitting (19,920 params eliminated)

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
        # 1. Load STRING_GNN with EXPANDED fine-tuning (4 layers)
        #    Unfreeze last 4 GNN message-passing layers + post_mp
        #    with tiered learning rates (near layers get higher LR)
        # ============================================================
        self.print("Loading STRING_GNN model (expanded fine-tuning: last 4 layers + post_mp)...")
        string_gnn = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        string_gnn.eval()

        # Freeze ALL STRING_GNN parameters first
        for param in string_gnn.parameters():
            param.requires_grad = False

        # Unfreeze last 4 message-passing layers + post_mp
        n_finetune = self.hparams.string_finetune_layers  # = 4
        total_layers = 8  # STRING_GNN has 8 GCN message-passing layers

        # Track parameter groups separately for tiered LR
        self._string_near_params: List[nn.Parameter] = []   # mps.6, mps.7 (closer to output)
        self._string_far_params: List[nn.Parameter] = []    # mps.4, mps.5 (further from output)
        self._string_postmp_params: List[nn.Parameter] = [] # post_mp

        n_near = 2  # Last 2 layers (mps.6, mps.7) get higher LR
        n_far = n_finetune - n_near  # Remaining layers (mps.4, mps.5) get lower LR

        def _matches_layer(param_name: str, layer_idx: int) -> bool:
            """Check if param_name belongs to mps.{layer_idx}."""
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

        # Cast unfrozen params to float32 for stable optimization
        for name, param in string_gnn.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        total_string = sum(p.numel() for p in string_gnn.parameters())
        self.print(
            f"STRING_GNN: {total_string:,} total params, "
            f"{unfrozen_params:,} trainable (last {n_finetune} GCN layers + post_mp)"
        )
        self.string_gnn = string_gnn

        # Load graph data (needed for forward pass)
        graph_data = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index = graph_data["edge_index"]
        edge_weight = graph_data.get("edge_weight", None)
        self.register_buffer("edge_index", edge_index.long())
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight.float())
        else:
            self.register_buffer("edge_weight", None)

        # ============================================================
        # 2. Load FROZEN ESM2-3B (KEY UPGRADE from parent's ESM2-650M)
        #    ESM2-3B produces 2560-dim embeddings (vs 1280-dim for 650M)
        #    Tree-best (F1=0.5283) uses ESM2-3B
        #    Memory: ~5.4GB per GPU in bf16, feasible on H100 80GB
        # ============================================================
        self.print("Loading FROZEN ESM2-3B (2560-dim, no LoRA, no fine-tuning)...")
        backbone = EsmForMaskedLM.from_pretrained(ESM2_MODEL, dtype=torch.bfloat16)

        # Freeze ALL ESM2 parameters
        for param in backbone.parameters():
            param.requires_grad = False

        # Enable gradient checkpointing for memory efficiency
        backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "config"):
            backbone.config.use_cache = False

        self.backbone = backbone
        total_esm2 = sum(p.numel() for p in self.backbone.parameters())
        self.print(f"ESM2-3B: {total_esm2:,} total params, all FROZEN")

        # ============================================================
        # 3. cond_emb projection + learnable gain + gated fusion
        #    + 3-block PreNorm MLP + flat head
        #    KEY CHANGES:
        #    - cond_proj: 2560 → 256 (ESM2-3B has 2560-dim)
        #    - GatedFusion protein_dim: 2560
        #    - head_dropout: 0.45 (increased from 0.30)
        #    - head_output_dropout: 0.15 (increased from 0.10)
        #    - per_gene_bias REMOVED (was overfitting-prone 19,920 params)
        # ============================================================
        # Project protein embedding to STRING dimension for cond_emb
        # ESM2-3B: 2560-dim → STRING: 256-dim
        self.cond_proj = nn.Linear(ESM2_DIM, STRING_DIM, bias=False)

        # Learnable scalar gain for STRING cond_emb (initialized to 1.0)
        self.string_gain = nn.Parameter(torch.ones(1))

        # Gated fusion: protein_emb (2560) + string_emb (256) → fused (FUSION_DIM=512)
        # protein_dim updated to ESM2_DIM=2560 for ESM2-3B
        self.gated_fusion = GatedFusion(
            protein_dim=ESM2_DIM,
            string_dim=STRING_DIM,
            fusion_dim=FUSION_DIM,
        )

        # Input projection: FUSION_DIM → HEAD_HIDDEN
        self.input_proj = nn.Linear(FUSION_DIM, self.hparams.head_hidden_dim, bias=True)

        # 3-block PreNorm residual MLP
        # KEY CHANGE: head_dropout increased from 0.30 → 0.45 for stronger regularization
        self.blocks = nn.ModuleList([
            PreNormResBlock(
                hidden_dim=self.hparams.head_hidden_dim,
                inner_dim=self.hparams.head_inner_dim,
                dropout=self.hparams.head_dropout,
            )
            for _ in range(self.hparams.n_blocks)
        ])

        # Output head
        self.output_norm = nn.LayerNorm(self.hparams.head_hidden_dim)
        # KEY CHANGE: head_output_dropout increased from 0.10 → 0.15
        self.head_dropout_layer = nn.Dropout(self.hparams.head_output_dropout)
        self.output_fc = nn.Linear(self.hparams.head_hidden_dim, N_GENES * N_CLASSES, bias=True)

        # KEY CHANGE: per_gene_bias REMOVED
        # Parent had per_gene_bias = nn.Parameter(torch.zeros(N_CLASSES, N_GENES))
        # Removing 19,920 overfitting-prone params as recommended in parent feedback

        # Cast trainable params to float32
        for mod in [self.cond_proj, self.gated_fusion, self.input_proj, self.blocks, self.output_norm, self.output_fc]:
            if mod is not None:
                for p in mod.parameters():
                    p.data = p.data.float()
        self.string_gain.data = self.string_gain.data.float()

        # ============================================================
        # 4. Muon parameter classification
        #    Muon: hidden weight matrices in PreNorm blocks (fc1, fc2)
        #    AdamW: everything else (norms, biases, head output, cond_proj, etc.)
        # ============================================================
        self._muon_hidden_weights = []
        self._adamw_non_hidden = []

        # Classify block parameters
        for block in self.blocks:
            for name, p in block.named_parameters():
                if p.requires_grad:
                    if p.ndim >= 2 and ("fc1.weight" in name or "fc2.weight" in name):
                        self._muon_hidden_weights.append(p)
                    else:
                        self._adamw_non_hidden.append(p)

        # Input projection hidden weight goes to Muon
        if self.input_proj.weight.ndim >= 2:
            self._muon_hidden_weights.append(self.input_proj.weight)
        if self.input_proj.bias is not None:
            self._adamw_non_hidden.append(self.input_proj.bias)

        # Output FC to AdamW (output layer, not for Muon)
        for p in self.output_fc.parameters():
            self._adamw_non_hidden.append(p)

        # All other params (cond_proj, string_gain, fusion, norms) → AdamW
        for mod in [self.cond_proj, self.gated_fusion, self.output_norm]:
            for p in mod.parameters():
                if p.requires_grad:
                    self._adamw_non_hidden.append(p)
        self._adamw_non_hidden.append(self.string_gain)
        # NOTE: per_gene_bias removed, so no longer appended here

        # ============================================================
        # 5. Loss: Weighted Cross-Entropy (no focal, no label smoothing)
        #    WCE is the proven loss for tree-best frozen ESM2 + STRING nodes
        #    Inverse-frequency class weights to handle 92.8% neutral imbalance
        # ============================================================
        freq = torch.tensor([0.9282, 0.0477, 0.0241], dtype=torch.float32)
        weights = 1.0 / freq
        weights = weights / weights.mean()
        self.register_buffer("class_weights", weights)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"FrozenESM2-3B + "
            f"STRING_GNN(4-layer ft) + "
            f"GatedFusion(dim={FUSION_DIM}) + "
            f"PreNorm-MLP(h={self.hparams.head_hidden_dim}x{self.hparams.n_blocks}) + "
            f"dropout={self.hparams.head_dropout} + "
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

        Args:
            protein_emb: [B, ESM2_DIM] protein embeddings (2560-dim for ESM2-3B)
            string_idx: [B] STRING node indices for the perturbed genes

        Returns:
            string_emb: [B, STRING_DIM] STRING embeddings for perturbed genes
        """
        n_nodes = self.hparams.n_string_nodes
        device = protein_emb.device

        # Project protein embedding to STRING dimension → cond_emb contribution
        # cond_proj: Linear(2560 → 256) for ESM2-3B (was 1280 → 256 for ESM2-650M)
        cond_per_sample = self.cond_proj(protein_emb.float())  # [B, 256]
        cond_per_sample = cond_per_sample * self.string_gain   # learnable gain

        # Build full node cond_emb [N, 256] using scatter_add (out-of-place for autograd)
        valid_mask = string_idx < n_nodes  # [B] bool

        cond_emb = torch.zeros(n_nodes, STRING_DIM, device=device, dtype=cond_per_sample.dtype)
        if valid_mask.any():
            valid_idx = string_idx[valid_mask]  # [n_valid]
            valid_cond = cond_per_sample[valid_mask]  # [n_valid, 256]
            cond_emb = cond_emb.index_add(0, valid_idx, valid_cond)

        # Run STRING_GNN forward with conditioning
        outputs = self.string_gnn(
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            cond_emb=cond_emb,
        )
        string_embs = outputs.last_hidden_state  # [n_nodes, 256]

        # Pad with zero row at index n_nodes for "not-in-STRING" fallback
        pad = torch.zeros(1, STRING_DIM, dtype=string_embs.dtype, device=device)
        string_embs_padded = torch.cat([string_embs, pad], dim=0)  # [n_nodes+1, 256]

        # Lookup per sample
        return string_embs_padded[string_idx]  # [B, 256]

    def _encode(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Encode a batch → [B, FUSION_DIM] via gated fusion."""
        # --- FROZEN ESM2-3B forward ---
        with torch.no_grad():  # No gradients for frozen ESM2-3B
            out = self.backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
        # Last hidden state: [B, T, 2560] for ESM2-3B
        hidden = out.hidden_states[-1].float()

        # Mean pooling over valid (non-padding, non-special) tokens
        attn_mask = batch["attention_mask"].float()  # [B, T], 1=valid
        seq_lens = attn_mask.sum(dim=1).long()  # [B] actual seq lengths including special
        B, T = attn_mask.shape
        positions = torch.arange(T, device=attn_mask.device).unsqueeze(0).expand(B, -1)  # [B, T]
        eos_pos = (seq_lens - 1).unsqueeze(1)  # [B, 1]
        special_mask = (positions == 0) | (positions == eos_pos)  # [B, T]
        valid_mask = attn_mask.bool() & ~special_mask  # [B, T]

        # Mean pool over valid tokens
        valid_float = valid_mask.float()
        protein_emb = (hidden * valid_float.unsqueeze(-1)).sum(dim=1)  # [B, 2560]
        count = valid_float.sum(dim=1, keepdim=True).clamp(min=1e-9)
        protein_emb = protein_emb / count  # [B, 2560]

        # --- STRING_GNN with cond_emb + expanded fine-tuning → [B, 256] ---
        string_emb = self._get_string_embeddings_with_cond(
            protein_emb,
            batch["string_idx"],
        )

        # --- Gated Fusion → [B, FUSION_DIM=512] ---
        return self.gated_fusion(protein_emb, string_emb)

    def _forward_head(self, fused_emb: torch.Tensor) -> torch.Tensor:
        """Apply 3-block PreNorm MLP + flat head → [B, N_CLASSES, N_GENES].

        KEY CHANGE: per_gene_bias removed (was 19,920 overfitting-prone params)
        """
        x = self.input_proj(fused_emb)   # [B, HEAD_HIDDEN]
        for block in self.blocks:
            x = block(x)                  # [B, HEAD_HIDDEN]
        x = self.output_norm(x)           # [B, HEAD_HIDDEN]
        x = self.head_dropout_layer(x)    # [B, HEAD_HIDDEN]
        out = self.output_fc(x)           # [B, N_GENES*N_CLASSES]
        out = out.view(-1, N_CLASSES, N_GENES)  # [B, N_CLASSES, N_GENES]
        return out  # No per_gene_bias addition (removed to reduce overfitting)

    def _compute_loss_hard(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """WCE loss with hard (integer) labels."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return wce_loss(logits_flat, labels_flat, self.class_weights)

    def _compute_loss_soft(self, logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
        """WCE loss with soft (Mixup-interpolated) labels.

        logits: [B, N_CLASSES, N_GENES]
        soft_labels: [B, N_GENES, N_CLASSES] soft one-hot labels
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        soft_flat = soft_labels.reshape(-1, N_CLASSES).float()
        return wce_loss_soft(logits_flat, soft_flat, self.class_weights)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        fused_emb = self._encode(batch)  # [B, FUSION_DIM]

        # Manifold Mixup in fused embedding space (prob=mixup_prob)
        apply_mixup = (
            self.hparams.mixup_prob > 0
            and torch.rand(1).item() < self.hparams.mixup_prob
            and "label" in batch
            and batch["label"].shape[0] >= 2  # Need at least 2 samples to mix
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
        fused_emb = self._encode(batch)  # No Mixup at validation
        logits = self._forward_head(fused_emb)
        loss = self._compute_loss_hard(logits, batch["label"])
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
        # sync_dist=False: F1 is already identical across ranks (computed from all_gathered data)
        # Using sync_dist=True here can cause NCCL shutdown timeouts in fast_dev_run mode
        self.log("val_f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        fused_emb = self._encode(batch)
        logits = self._forward_head(fused_emb)   # [B, 3, 6640]
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
            # Gather string lists via pickle + CUDA byte tensors
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

        # Optional: gather test labels if available
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

        if self.trainer.is_global_zero:
            n_preds = all_preds.shape[0]
            n_ids = len(all_pert_ids)
            min_len = min(n_preds, n_ids)

            # Deduplicate by pert_id (keep first occurrence)
            seen = set()
            unique_idx = []
            for i, pid in enumerate(all_pert_ids[:min_len]):
                if pid not in seen:
                    seen.add(pid)
                    unique_idx.append(i)

            dedup_preds = all_preds[unique_idx]
            dedup_ids = [all_pert_ids[i] for i in unique_idx]
            dedup_syms = [all_symbols[i] for i in unique_idx]

            _save_test_predictions(
                pert_ids=dedup_ids,
                symbols=dedup_syms,
                preds=dedup_preds.float().cpu().numpy(),
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

            if has_labels and all_labels is not None:
                dedup_labels = all_labels[unique_idx]
                f1 = _compute_per_gene_f1(
                    dedup_preds.float().cpu().numpy(),
                    dedup_labels.cpu().numpy(),
                )
                # sync_dist=False: F1 is already gathered and identical across ranks
                self.log("test_f1", f1, prog_bar=True, sync_dist=False)
                self.print(f"Test F1: {f1:.4f}")

    def configure_optimizers(self):
        """Muon+AdamW optimizer with CosineWarmRestarts + warmup.

        KEY CHANGE: weight_decay increased from 8e-4 → 2e-3 for stronger L2 regularization.

        Param groups:
        - Muon: hidden weight matrices (fc1.weight, fc2.weight in PreNorm blocks, input_proj.weight)
        - AdamW: all other trainable params (norms, biases, cond_proj, gated_fusion, string gain, head output)
        - AdamW (STRING near): mps.6, mps.7 + post_mp → lr=lr_string_near (5e-5)
        - AdamW (STRING far): mps.4, mps.5 → lr=lr_string_far (3e-5)

        Muon is NOT used for STRING_GNN (fine-tuned) — Muon is only for MLP hidden weights.
        Muon is NOT used for ESM2 (frozen).
        """
        from muon import MuonWithAuxAdam

        string_near_params = self._string_near_params + self._string_postmp_params
        string_far_params = self._string_far_params

        # Build parameter groups
        param_groups = [
            # Muon: hidden weight matrices of PreNorm blocks + input_proj
            dict(
                params=self._muon_hidden_weights,
                use_muon=True,
                lr=self.hparams.lr_muon,
                weight_decay=self.hparams.weight_decay,
                momentum=0.95,
            ),
            # AdamW: non-hidden params (head output, norms, biases, cond_proj, fusion, etc.)
            dict(
                params=self._adamw_non_hidden,
                use_muon=False,
                lr=self.hparams.lr_adamw_head,
                betas=(0.9, 0.95),
                weight_decay=self.hparams.weight_decay,
            ),
        ]

        # STRING near params (AdamW at higher LR)
        if string_near_params:
            param_groups.append(dict(
                params=string_near_params,
                use_muon=False,
                lr=self.hparams.lr_string_near,
                betas=(0.9, 0.95),
                weight_decay=self.hparams.weight_decay,
            ))

        # STRING far params (AdamW at lower LR)
        if string_far_params:
            param_groups.append(dict(
                params=string_far_params,
                use_muon=False,
                lr=self.hparams.lr_string_far,
                betas=(0.9, 0.95),
                weight_decay=self.hparams.weight_decay,
            ))

        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        optimizer = MuonWithAuxAdam(param_groups)

        # CosineAnnealingWarmRestarts + linear warmup via LambdaLR wrapper
        # T_0=100, T_mult=2 → warm restarts at epoch 100, 300 (within 400-epoch budget)
        # Using LambdaLR wrapper to avoid CosineAnnealingWarmRestarts Muon incompatibility
        T_0 = self.hparams.t0_epochs
        T_mult = self.hparams.t_mult
        warmup_epochs = self.hparams.warmup_epochs

        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < warmup_epochs:
                return float(current_epoch) / float(max(1, warmup_epochs))

            # After warmup: CosineAnnealingWarmRestarts logic
            epoch = current_epoch - warmup_epochs
            t_cur = epoch
            t_i = T_0
            while t_cur >= t_i:
                t_cur -= t_i
                t_i = int(t_i * T_mult)

            # Cosine decay within the current cycle
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
    """Per-gene macro F1, averaged over all n_genes (matches calc_metric.py).

    Args:
        preds: [n_samples, 3, n_genes] — class logits / probabilities
        labels: [n_samples, n_genes] — integer class labels in {0, 1, 2}
    Returns:
        scalar F1
    """
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
    top_k: int = 5,
) -> None:
    """Load top-K checkpoints, average their logit predictions, and save.

    Multi-GPU compatible: Runs on single GPU (rank 0 after DDP training completes).
    Sort checkpoints by val_f1 extracted from filename, take top-K.
    Average logit predictions across checkpoints before writing test_predictions.tsv.

    Uses rglob("*.ckpt") to handle checkpoints in subdirectories.
    """
    # Use rglob to handle checkpoints saved in subdirectories
    ckpt_files = sorted(
        [f for f in checkpoint_dir.rglob("*.ckpt") if "last" not in f.name],
        key=lambda x: x.stat().st_mtime,
        reverse=False,
    )

    def extract_f1(f: Path) -> float:
        name = f.stem
        # Pattern: best-ep=xxx-f1=0.xxxx  OR  epoch=xxx-step=xxx-val_f1=0.xxxx
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

    # Move model to single GPU for ensemble inference
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Collect predictions from each checkpoint
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

        preds_arr = np.concatenate(preds_list, axis=0)  # [n_test, 3, 6640]
        all_ensemble_preds.append(preds_arr)

        if all_pert_ids is None:
            all_pert_ids = pert_ids_list
            all_symbols = symbols_list

    # Average logits across checkpoints
    avg_preds = np.mean(all_ensemble_preds, axis=0)  # [n_test, 3, 6640]

    if all_pert_ids is not None:
        # Deduplicate
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
            "Node2-1-2-1-1-1-1-1: Frozen ESM2-3B + STRING_GNN (4-layer ft) + "
            "Muon + GatedFusion(512) + PreNorm MLP (h=384, dropout=0.45) + "
            "WD=2e-3 + no per_gene_bias (aggressive regularization)"
        )
    )
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=400)
    p.add_argument("--lr-muon", type=float, default=0.01)
    p.add_argument("--lr-adamw-head", type=float, default=3e-4)
    p.add_argument("--lr-string-near", type=float, default=5e-5)
    p.add_argument("--lr-string-far", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=2e-3)       # KEY: 8e-4 → 2e-3
    p.add_argument("--head-hidden-dim", type=int, default=HEAD_HIDDEN)
    p.add_argument("--head-inner-dim", type=int, default=HEAD_HIDDEN * 2)
    p.add_argument("--head-dropout", type=float, default=0.45)        # KEY: 0.30 → 0.45
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--head-output-dropout", type=float, default=0.15) # KEY: 0.10 → 0.15
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--early-stop-patience", type=int, default=50)     # KEY: 80 → 50
    p.add_argument("--string-finetune-layers", type=int, default=4)
    p.add_argument("--mixup-alpha", type=float, default=0.3)           # KEY: 0.2 → 0.3
    p.add_argument("--mixup-prob", type=float, default=0.65)
    p.add_argument("--t0-epochs", type=int, default=100)
    p.add_argument("--t-mult", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
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

    # Get n_string_nodes from STRING_GNN config
    _node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    n_string_nodes = len(_node_names)

    model = PerturbModule(
        head_hidden_dim=args.head_hidden_dim,
        head_inner_dim=args.head_inner_dim,
        head_dropout=args.head_dropout,
        n_blocks=args.n_blocks,
        head_output_dropout=args.head_output_dropout,
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
        # Debug: limited training/val batches, but FULL test dataset for comprehensive testing
        limit_train = limit_val = args.debug_max_step
        limit_test = 1.0  # Use full test dataset for debugging
        max_steps = args.debug_max_step
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-ep={epoch:03d}-f1={val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=5,   # Save top-5 for ensemble
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stop_patience,   # KEY: reduced to 50 from 80
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
        strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=1.0,
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
        # In debug mode, use current model weights without checkpoint loading
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        # Step 1: Run test with best checkpoint to get initial result
        best_ckpt = checkpoint_cb.best_model_path
        if not best_ckpt or not Path(best_ckpt).exists():
            best_ckpt = str(output_dir / "checkpoints" / "last.ckpt")
        print(f"Test checkpoint path (best): {best_ckpt}")
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path=best_ckpt)

        # Step 2: Checkpoint ensemble — works for both single-GPU and multi-GPU
        use_ensemble = not args.no_ensemble
        checkpoint_dir = output_dir / "checkpoints"

        if use_ensemble and checkpoint_dir.exists() and trainer.is_global_zero:
            print("\nRunning checkpoint ensemble for improved test predictions...")
            _ensemble_checkpoints_and_test(
                model=model,
                datamodule=datamodule,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                top_k=5,
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
        # Write plain numeric value for FeedbackAgent compatibility
        score_path.write_text(str(score_value))
        print(f"Test score → {score_path}: {score_value}")


if __name__ == "__main__":
    main()
