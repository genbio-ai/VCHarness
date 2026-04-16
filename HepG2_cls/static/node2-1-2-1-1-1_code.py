"""Node 2-1-2-1-1-1: ESM2-650M (LoRA r=24) + STRING_GNN (partial fine-tune, last 4 layers)
         + Gated Fusion + Wider Head + CosineWarmRestarts + Manifold Mixup for HepG2 DEG Prediction.

Key improvements over parent node2-1-2-1-1 (Test F1=0.4724):

  1. STRING_GNN expanded fine-tuning (2 → 4 layers) [HIGHEST PRIORITY]
     - Feedback explicitly identified this as the primary bottleneck
     - Every node with F1>0.47 benefits from STRING fine-tuning; broader scope = better signal
     - Tiered LRs: mps.4-5 get lr=3e-5, mps.6-7+post_mp get lr=5e-5

  2. Wider head (512 → 640) with higher dropout (0.40 → 0.45)
     - Feedback: 512-dim may be underdimensioned for 256→19920 mapping
     - Higher dropout compensates for the wider capacity to prevent overfitting
     - Still much smaller than the parent's 1280 (which caused severe overfitting)

  3. CosineAnnealingWarmRestarts (T_0=80, T_mult=2) instead of single cosine cycle
     - Proven in tree: node3-3-1-2-1-1-1 (F1=0.5243), node1-3-2-2-1-1-1-1 (F1=0.4914)
     - Warm restarts help escape local optima that the parent converged to at epoch 113
     - T_0=80, T_mult=2 → warm restarts at 80, 240, 560 epochs
     - With max_epochs=250, gets 2 warm restart cycles

  4. Focal gamma increased from 1.5 to 2.0 (without Mixup interference)
     - Parent used γ=1.5 to counterbalance Mixup interactions (Mixup was removed)
     - Since Mixup is now managed (Manifold Mixup in fused space), γ=2.0 is appropriate
     - Confirmed beneficial in tree: node3-3-1-1-1-1-1 (F1=0.4851), node4-1-1-1-1-1 (F1=0.5175)

  5. Manifold Mixup in fused embedding space (prob=0.5, alpha=0.2)
     - Doubles effective training set through embedding interpolation
     - Applied in the 256-dim fused representation space (not in input or label space)
     - Prevents overfitting on the very small training set (1,273 samples)
     - Proven: node3-3-1-2-1-1-1 (F1=0.5243) used Manifold Mixup in fusion space
     - Note: Applied AFTER gated fusion, on the 256-dim fused representation

  6. Longer training (max_epochs=250, patience=50)
     - Parent was still improving at epoch 153 with patience=40
     - With warm restarts, training needs more epochs to explore multiple cycles
     - Matches the node3-3-1-2-1-1-1 recipe (best-in-tree: 0.5243)

  7. Improved checkpoint ensemble: top-7 checkpoints across warm restart cycles
     - More diverse than top-5 since checkpoints span different LR phases/restarts

Memory rationale:
  - node2-1-2-1-1 (parent, F1=0.4724): STRING 2-layer fine-tune → bottleneck identified
  - node4-1-1-1-1-1 (F1=0.5175): ESM2 LoRA + STRING cond_emb + focal + Mixup
  - node3-3-1-2-1-1-1 (F1=0.5243): frozen ESM2+STRING + CosineWarmRestarts + Manifold Mixup
  - node3-1-1-1-1-2 (F1=0.5243): frozen ESM2-650M + STRING + gated fusion → exact recipe
  - node3-3-1-1-1-1-1 (F1=0.4851): focal γ=2.0 confirmed effective for STRING models without Mixup
  - node1-3-2-2-1-1-1-1 (F1=0.4914): top-5 ensemble +0.0035 gain

Auxiliary data dependencies:
  - Protein FASTA: /home/data/genome/hg38_gencode_protein.fa (genomic-data-skill)
  - STRING_GNN:    /home/Models/STRING_GNN (string-gnn-model-skill)
  - ESM2-650M:     facebook/esm2_t33_650M_UR50D (esm2-protein-model-skill)
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
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, EsmForMaskedLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
PROTEIN_FASTA = "/home/data/genome/hg38_gencode_protein.fa"
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")

N_GENES = 6640
N_CLASSES = 3
MAX_SEQ_LEN = 512       # ESM2 max context, enough for most human proteins
ESM2_DIM = 1280         # ESM2-650M hidden size
STRING_DIM = 256        # STRING_GNN output dimension
PROJ_DIM = 256          # Fusion dimension
HEAD_HIDDEN = 640       # MLP head hidden dimension (wider than parent's 512, narrower than grandparent's 1280)
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
# Gated Fusion Module
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Gated fusion of protein and STRING_GNN embeddings.

    Protein (1280-dim) is projected to 256-dim (same as STRING_GNN).
    A gate vector learned from the concatenation selects between the two modalities.
    Output: [B, 256] fused representation.

    Proven better than naive concatenation in node4 lineage (test F1=0.5175).
    """

    def __init__(self, protein_dim: int = ESM2_DIM, string_dim: int = STRING_DIM, fusion_dim: int = PROJ_DIM) -> None:
        super().__init__()
        self.protein_proj = nn.Linear(protein_dim, fusion_dim, bias=True)
        self.protein_norm = nn.LayerNorm(fusion_dim)
        self.string_norm = nn.LayerNorm(string_dim)
        # Gate learns which modality to trust per dimension
        self.gate_fc = nn.Linear(fusion_dim * 2, fusion_dim, bias=True)
        nn.init.zeros_(self.gate_fc.bias)

    def forward(self, protein_emb: torch.Tensor, string_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            protein_emb: [B, ESM2_DIM] from mean pooling
            string_emb: [B, STRING_DIM] from STRING_GNN lookup
        Returns:
            fused: [B, PROJ_DIM]
        """
        p = self.protein_norm(self.protein_proj(protein_emb.float()))  # [B, 256]
        s = self.string_norm(string_emb.float())  # [B, 256]
        concat = torch.cat([p, s], dim=-1)  # [B, 512]
        gate = torch.sigmoid(self.gate_fc(concat))  # [B, 256]
        return gate * p + (1.0 - gate) * s  # [B, 256]


# ---------------------------------------------------------------------------
# Prediction Head
# ---------------------------------------------------------------------------
class PerturbHead(nn.Module):
    """MLP head: [B, in_dim] → [B, 3, N_GENES].

    2-layer MLP: LN → Linear → GELU → Dropout → Linear(→19920) → Reshape
    + per-gene bias [3, 6640] added to output logits.

    Wider than parent's 512 (640 here) to address potential underdimensioning
    while staying below grandparent's 1280 (which caused severe overfitting).
    Higher dropout (0.45) compensates for the increased capacity.
    """

    def __init__(
        self,
        in_dim: int = PROJ_DIM,
        hidden_dim: int = HEAD_HIDDEN,
        n_genes: int = N_GENES,
        dropout: float = 0.45,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes * N_CLASSES),
        )
        # Per-gene bias: learnable global expression tendencies per gene
        self.per_gene_bias = nn.Parameter(torch.zeros(N_CLASSES, n_genes))
        self.n_genes = n_genes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)  # [B, n_genes * 3]
        out = out.view(-1, N_CLASSES, self.n_genes)  # [B, 3, 6640]
        return out + self.per_gene_bias.unsqueeze(0)  # broadcast [B, 3, 6640]


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """Focal cross-entropy loss with class weights and label smoothing.

    Args:
        logits: [B*N_GENES, N_CLASSES] float32 logits
        labels: [B*N_GENES] long labels in {0, 1, 2}
        class_weights: [N_CLASSES] float32 per-class weights
        gamma: focal loss exponent (2.0 for stronger minority-class emphasis)
        label_smoothing: label smoothing epsilon
    """
    num_classes = logits.shape[1]

    # Smooth labels
    with torch.no_grad():
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        smooth_labels = one_hot * (1.0 - label_smoothing) + label_smoothing / num_classes

    # Log-softmax
    log_probs = F.log_softmax(logits, dim=-1)  # [B, C]
    probs = torch.exp(log_probs)  # [B, C]

    # Focal weight: (1-p_t)^gamma where p_t is the probability of the true class
    p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
    focal_weight = (1.0 - p_t.detach()).pow(gamma)  # [B]

    # Cross entropy with smooth labels
    ce = -(smooth_labels * log_probs).sum(dim=-1)  # [B]

    # Class weights: weighted by true class
    if class_weights is not None:
        cw = class_weights[labels]  # [B]
        ce = ce * cw

    return (focal_weight * ce).mean()


def focal_loss_soft(
    logits: torch.Tensor,
    soft_labels: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal cross-entropy loss with soft (Mixup-interpolated) labels.

    Used for Manifold Mixup where labels are interpolated between two samples.

    Args:
        logits: [B*N_GENES, N_CLASSES] float32 logits
        soft_labels: [B*N_GENES, N_CLASSES] float32 soft class probabilities
        class_weights: [N_CLASSES] float32 per-class weights
        gamma: focal loss exponent
    """
    log_probs = F.log_softmax(logits, dim=-1)  # [B, C]
    probs = torch.exp(log_probs)  # [B, C]

    # Focal weight: based on max probability (most confident class)
    p_max = probs.max(dim=-1).values  # [B]
    focal_weight = (1.0 - p_max.detach()).pow(gamma)  # [B]

    # Soft cross-entropy
    ce = -(soft_labels * log_probs).sum(dim=-1)  # [B]

    # Apply class weights: use expected weight from soft labels
    if class_weights is not None:
        # Weighted by expected class (expectation over soft distribution)
        expected_cw = (soft_labels * class_weights.unsqueeze(0)).sum(dim=-1)  # [B]
        ce = ce * expected_cw

    return (focal_weight * ce).mean()


# ---------------------------------------------------------------------------
# Manifold Mixup Helper
# ---------------------------------------------------------------------------
def manifold_mixup(
    fused_emb: torch.Tensor,
    labels: Optional[torch.Tensor],
    alpha: float = 0.2,
    training: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
    """Apply Manifold Mixup in the fused embedding space.

    Interpolates between pairs of samples in the 256-dim fused representation.
    This provides strong regularization without corrupting the structured
    protein + PPI signal that the encoder produces.

    Args:
        fused_emb: [B, PROJ_DIM] fused embeddings
        labels: [B, N_GENES] integer class labels in {0, 1, 2}, or None for test
        alpha: Dirichlet concentration parameter (0.2 recommended)
        training: whether to apply Mixup (disabled at eval/test time)

    Returns:
        mixed_emb: [B, PROJ_DIM] mixed embeddings
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
    # labels: [B, N_GENES] in {0,1,2}
    one_hot_a = F.one_hot(labels, num_classes=N_CLASSES).float()  # [B, N_GENES, 3]
    one_hot_b = F.one_hot(labels[perm], num_classes=N_CLASSES).float()  # [B, N_GENES, 3]
    mixed_labels = lam * one_hot_a + (1.0 - lam) * one_hot_b  # [B, N_GENES, 3]

    return mixed_emb, mixed_labels, lam


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 24,
        lora_alpha: int = 48,
        lora_dropout: float = 0.05,
        head_hidden_dim: int = HEAD_HIDDEN,
        head_dropout: float = 0.45,
        lr_protein: float = 5e-5,
        lr_string: float = 1e-4,
        lr_string_finetune_near: float = 5e-5,   # lr for last 2 unfrozen STRING layers (mps.6,7)
        lr_string_finetune_far: float = 3e-5,    # lr for 2 additional layers (mps.4,5) - tiered
        weight_decay: float = 5e-4,
        max_epochs: int = 250,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        warmup_epochs: int = 10,
        n_string_nodes: int = 18870,
        # STRING fine-tuning: unfreeze last N GNN layers
        string_finetune_layers: int = 4,  # Expanded from parent's 2 → 4
        # Manifold Mixup
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.5,
        # CosineWarmRestarts
        t0_epochs: int = 80,  # First cycle length
        t_mult: int = 2,      # Cycle length multiplier
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone: Optional[nn.Module] = None
        self.string_gnn: Optional[nn.Module] = None
        self.cond_proj: Optional[nn.Linear] = None
        self.string_gain: Optional[nn.Parameter] = None
        self.gated_fusion: Optional[GatedFusion] = None
        self.head: Optional[PerturbHead] = None

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # ============================================================
        # 1. Load STRING_GNN with EXPANDED fine-tuning (4 layers)
        #    Unfreeze last 4 GNN message-passing layers + post_mp
        #    with tiered learning rates (near layers get higher LR)
        #
        #    KEY IMPROVEMENT: Parent only unfroze 2 layers → bottleneck confirmed
        #    by feedback. Every node achieving >0.47 needs STRING fine-tuning;
        #    expanding to 4 layers follows the recommendation from feedback.md
        # ============================================================
        self.print("Loading STRING_GNN model (expanded fine-tuning: last 4 layers + post_mp)...")
        string_gnn = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        string_gnn.eval()  # Start in eval mode; will switch to train mode during training

        # Freeze ALL STRING_GNN parameters first
        for param in string_gnn.parameters():
            param.requires_grad = False

        # Unfreeze last 4 message-passing layers + post_mp
        # STRING_GNN architecture: mps.0 through mps.7 (8 GCN layers), then post_mp
        n_finetune = self.hparams.string_finetune_layers  # = 4
        total_layers = 8  # STRING_GNN has 8 GCN message-passing layers
        # For n_finetune=4: unfreeze mps.4, mps.5, mps.6, mps.7 + post_mp

        # Track parameter groups separately for tiered LR
        self._string_near_params: List[nn.Parameter] = []   # mps.6, mps.7 (closer to output)
        self._string_far_params: List[nn.Parameter] = []    # mps.4, mps.5 (further from output)
        self._string_postmp_params: List[nn.Parameter] = [] # post_mp

        n_near = 2  # Last 2 layers (mps.6, mps.7) get higher LR
        n_far = n_finetune - n_near  # Remaining layers (mps.4, mps.5) get lower LR

        def _matches_layer(param_name: str, layer_idx: int) -> bool:
            """Check if param_name belongs to mps.{layer_idx} (exact match, no partial digits)."""
            prefix = f"mps.{layer_idx}."
            return prefix in param_name or param_name.endswith(f"mps.{layer_idx}")

        unfrozen_params = 0
        for name, param in string_gnn.named_parameters():
            # Check near layers (mps.6, mps.7) — higher LR group
            is_near = any(_matches_layer(name, idx) for idx in range(total_layers - n_near, total_layers))
            # Check far layers (mps.4, mps.5) — lower LR group
            is_far = any(_matches_layer(name, idx) for idx in range(total_layers - n_finetune, total_layers - n_near))
            # Check post_mp
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
            f"{unfrozen_params:,} trainable (last {n_finetune} GCN layers + post_mp); "
            f"near(mps.6,7)={len(self._string_near_params)} tensors, "
            f"far(mps.4,5)={len(self._string_far_params)} tensors, "
            f"postmp={len(self._string_postmp_params)} tensors"
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
        # 2. Load ESM2-650M with LoRA (r=24)
        #    Target modules: query, key, value, dense
        #    Mean pooling over valid (non-special) tokens
        # ============================================================
        self.print(f"Loading ESM2-650M with LoRA r={self.hparams.lora_r}...")
        backbone = EsmForMaskedLM.from_pretrained(ESM2_MODEL, dtype=torch.bfloat16)
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            target_modules=["query", "key", "value", "dense"],
            layers_to_transform=None,  # All 33 attention layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable()
        if hasattr(self.backbone, "config"):
            self.backbone.config.use_cache = False

        # Cast trainable (LoRA) params to float32 for stable optimization
        for _name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ============================================================
        # 3. cond_emb projection + learnable gain + gated fusion + head
        # ============================================================
        # Project protein embedding to STRING dimension for cond_emb
        self.cond_proj = nn.Linear(ESM2_DIM, STRING_DIM, bias=False)

        # Learnable scalar gain for STRING cond_emb (initialized to 1.0)
        self.string_gain = nn.Parameter(torch.ones(1))

        # Gated fusion: protein_emb (1280) + string_emb (256) → fused (256)
        self.gated_fusion = GatedFusion(
            protein_dim=ESM2_DIM,
            string_dim=STRING_DIM,
            fusion_dim=PROJ_DIM,
        )

        # Prediction head with per-gene bias
        # head_hidden_dim=640 (wider than parent's 512, smaller than grandparent's 1280)
        self.head = PerturbHead(
            in_dim=PROJ_DIM,
            hidden_dim=self.hparams.head_hidden_dim,
            dropout=self.hparams.head_dropout,
        )

        # ============================================================
        # 4. Loss: Focal loss + class frequency weights + label smoothing
        #    Class frequencies from dataset: [class0: 92.82%, class-1: 4.77%, class+1: 2.41%]
        #    gamma=2.0 (restored from parent's 1.5; since Mixup now properly applied,
        #    γ=2.0 provides strong minority class emphasis without over-focusing)
        # ============================================================
        freq = torch.tensor([0.9282, 0.0477, 0.0241], dtype=torch.float32)
        # Inverse-frequency weights
        weights = 1.0 / freq
        weights = weights / weights.mean()
        self.register_buffer("class_weights", weights)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"ESM2-650M LoRA(r={self.hparams.lora_r}) + "
            f"Expanded STRING_GNN(last {self.hparams.string_finetune_layers} layers + post_mp) + "
            f"GatedFusion + PerGeneBias + ManifoldMixup | "
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
            protein_emb: [B, ESM2_DIM] protein embeddings
            string_idx: [B] STRING node indices for the perturbed genes

        Returns:
            string_emb: [B, STRING_DIM] STRING embeddings for perturbed genes
        """
        n_nodes = self.hparams.n_string_nodes
        device = protein_emb.device

        # Project protein embedding to STRING dimension → cond_emb contribution
        cond_per_sample = self.cond_proj(protein_emb.float())  # [B, 256]
        cond_per_sample = cond_per_sample * self.string_gain  # learnable gain

        # Build full node cond_emb [N, 256] using scatter_add (out-of-place for autograd)
        valid_mask = string_idx < n_nodes  # [B] bool

        cond_emb = torch.zeros(n_nodes, STRING_DIM, device=device, dtype=cond_per_sample.dtype)
        if valid_mask.any():
            valid_idx = string_idx[valid_mask]  # [n_valid]
            valid_cond = cond_per_sample[valid_mask]  # [n_valid, 256]
            cond_emb = cond_emb.index_add(0, valid_idx, valid_cond)

        # Run STRING_GNN forward with conditioning
        # Gradients flow through unfrozen layers (mps.4, mps.5, mps.6, mps.7, post_mp)
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
        """Encode a batch → [B, PROJ_DIM=256] via gated fusion."""
        # --- ESM2-650M forward ---
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        # Last hidden state: [B, T, 1280]
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
        protein_emb = (hidden * valid_float.unsqueeze(-1)).sum(dim=1)  # [B, 1280]
        count = valid_float.sum(dim=1, keepdim=True).clamp(min=1e-9)
        protein_emb = protein_emb / count  # [B, 1280]

        # --- STRING_GNN with cond_emb + expanded fine-tuning → [B, 256] ---
        string_emb = self._get_string_embeddings_with_cond(
            protein_emb,
            batch["string_idx"],
        )

        # --- Gated Fusion → [B, 256] ---
        return self.gated_fusion(protein_emb, string_emb)

    def _compute_loss_hard(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Focal loss with hard (integer) labels."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return focal_loss(
            logits_flat, labels_flat, self.class_weights,
            gamma=self.hparams.focal_gamma,
            label_smoothing=self.hparams.label_smoothing,
        )

    def _compute_loss_soft(
        self,
        logits: torch.Tensor,
        soft_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Focal loss with soft (Mixup-interpolated) labels.

        logits: [B, 3, N_GENES]
        soft_labels: [B, N_GENES, 3] soft one-hot labels
        """
        # Reshape logits: [B, 3, N_GENES] → [B*N_GENES, 3]
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        # Reshape soft_labels: [B, N_GENES, 3] → [B*N_GENES, 3]
        soft_flat = soft_labels.reshape(-1, N_CLASSES).float()
        return focal_loss_soft(
            logits_flat, soft_flat, self.class_weights,
            gamma=self.hparams.focal_gamma,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        fused_emb = self._encode(batch)  # [B, 256]

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
            logits = self.head(mixed_emb)
            if mixed_labels is not None:
                loss = self._compute_loss_soft(logits, mixed_labels)
            else:
                loss = self._compute_loss_hard(logits, batch["label"])
        else:
            logits = self.head(fused_emb)
            loss = self._compute_loss_hard(logits, batch["label"])

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        fused_emb = self._encode(batch)  # No Mixup at validation
        logits = self.head(fused_emb)
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
        # All ranks log the same val_f1 for consistent checkpoint naming
        self.log("val_f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        fused_emb = self._encode(batch)
        logits = self.head(fused_emb)  # [B, 3, 6640]
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
                self.log("test_f1", f1, prog_bar=True, sync_dist=False)
                self.print(f"Test F1: {f1:.4f}")

    def configure_optimizers(self):
        """Differential learning rate optimizer groups with CosineWarmRestarts + warmup.

        Param groups:
        - ESM2-650M LoRA adapters: lr=lr_protein (5e-5)
        - STRING_GNN near layers (mps.6, mps.7 + post_mp): lr=lr_string_finetune_near (5e-5)
        - STRING_GNN far layers (mps.4, mps.5): lr=lr_string_finetune_far (3e-5)
        - cond_proj, string_gain, gated_fusion, head: lr=lr_string (1e-4)

        All groups share weight_decay=5e-4.
        CosineAnnealingWarmRestarts with T_0=80, T_mult=2.
        """
        lora_params = [p for _, p in self.backbone.named_parameters() if p.requires_grad]

        # STRING_GNN trainable params, split by tiered LR groups
        string_near_params = self._string_near_params + self._string_postmp_params
        string_far_params = self._string_far_params

        # All other new params: cond_proj, string_gain, gated_fusion, head
        other_params = (
            list(self.cond_proj.parameters())
            + [self.string_gain]
            + list(self.gated_fusion.parameters())
            + list(self.head.parameters())
        )

        param_groups = [
            {
                "params": lora_params,
                "lr": self.hparams.lr_protein,
                "weight_decay": self.hparams.weight_decay,
                "name": "lora",
            },
            {
                "params": string_near_params,
                "lr": self.hparams.lr_string_finetune_near,
                "weight_decay": self.hparams.weight_decay,
                "name": "string_near",
            },
            {
                "params": other_params,
                "lr": self.hparams.lr_string,
                "weight_decay": self.hparams.weight_decay,
                "name": "head",
            },
        ]

        # Only include far params group if there are actually far parameters
        if string_far_params:
            param_groups.insert(2, {
                "params": string_far_params,
                "lr": self.hparams.lr_string_finetune_far,
                "weight_decay": self.hparams.weight_decay,
                "name": "string_far",
            })

        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        optimizer = torch.optim.AdamW(param_groups)

        # CosineAnnealingWarmRestarts + linear warmup
        # T_0=80, T_mult=2 → warm restarts at epoch 80, 240 (within 250 epoch budget)
        T_0 = self.hparams.t0_epochs
        T_mult = self.hparams.t_mult
        warmup_epochs = self.hparams.warmup_epochs

        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < warmup_epochs:
                return float(current_epoch) / float(max(1, warmup_epochs))

            # After warmup: CosineAnnealingWarmRestarts logic
            # Compute cycle and position within cycle
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
    top_k: int = 7,
) -> None:
    """Load top-K checkpoints, average their logit predictions, and save.

    Multi-GPU compatible: Runs on single GPU (rank 0 after DDP training completes).
    Sort checkpoints by val_f1 extracted from filename, take top-K.
    Average logit predictions across checkpoints before writing test_predictions.tsv.

    Uses top_k=7 to capture more diversity across warm restart cycles.
    """
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
                logits = model.head(fused_emb)
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
        description="Node2-1-2-1-1-1: ESM2-650M LoRA + Expanded STRING_GNN (4 layers) + Gated Fusion + ManifoldMixup + CosineWarmRestarts"
    )
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=250)
    p.add_argument("--lr-protein", type=float, default=5e-5)
    p.add_argument("--lr-string", type=float, default=1e-4)
    p.add_argument("--lr-string-finetune-near", type=float, default=5e-5)
    p.add_argument("--lr-string-finetune-far", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--lora-r", type=int, default=24)
    p.add_argument("--lora-alpha", type=int, default=48)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--head-hidden-dim", type=int, default=HEAD_HIDDEN)
    p.add_argument("--head-dropout", type=float, default=0.45)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--early-stop-patience", type=int, default=50)
    p.add_argument("--string-finetune-layers", type=int, default=4)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--t0-epochs", type=int, default=80)
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
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        lr_protein=args.lr_protein,
        lr_string=args.lr_string,
        lr_string_finetune_near=args.lr_string_finetune_near,
        lr_string_finetune_far=args.lr_string_finetune_far,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
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
        save_top_k=7,   # Save top-7 for ensemble (more diversity across warm restart cycles)
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stop_patience,
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
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120)),
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
        # After DDP training, all ranks have access to saved checkpoints.
        # Rank 0 runs single-GPU ensemble inference.
        use_ensemble = not args.no_ensemble
        checkpoint_dir = output_dir / "checkpoints"

        if use_ensemble and checkpoint_dir.exists() and trainer.is_global_zero:
            print("\nRunning checkpoint ensemble for improved test predictions...")
            _ensemble_checkpoints_and_test(
                model=model,
                datamodule=datamodule,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                top_k=7,
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
