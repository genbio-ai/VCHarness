"""Node 1-2-2-3: Frozen ESM2-650M + STRING_GNN Dual-Branch Architecture
               + Gated Sigmoidal Fusion + 3-Block Pre-Norm MLP (h=384)
               + Muon Optimizer + Focal Loss (gamma=2.0)
               + CosineAnnealingWarmRestarts (T_0=80, T_mult=2)
               + Manifold Mixup (prob=0.65) + Per-Gene Bias (wd=0.05)
================================================================
Node: node1-2-2-3
Parent: node1-2-2 (test F1=0.4433, over-regularized STRING-only)
Siblings:
  - node1-2-2-1 (F1=0.4582, STRING-only, MLP wd=0.03, gene_bias wd=0.10, Mixup 0.25)
  - node1-2-2-2 (F1=0.4571, STRING-only, MLP wd=0.01, gene_bias wd=0.05, Top-3 ensemble)

KEY DESIGN RATIONALE:
The node1-2-2-x sub-lineage has plateaued at ~0.458 F1. Both siblings independently
confirmed this ceiling through regularization tuning. Sibling feedback explicitly
recommends abandoning the STRING-only branch and transitioning to an ESM2+STRING
dual-branch architecture, which achieves F1=0.52+ in the node3-3 lineage.

This node implements the PROVEN TREE-BEST DUAL-BRANCH ARCHITECTURE from:
- node3-3-1-2-1-1-1 (F1=0.5243, tree best): frozen ESM2-650M + STRING_GNN,
  gated fusion, h=384, Muon+CosineWR(T_0=80,T_mult=2), Mixup prob=0.65
- node3-3-1-2-1 (F1=0.5170): frozen ESM2-150M + STRING_GNN, same recipe

ESM2-650M IMPLEMENTATION DETAILS:
- Protein sequences sourced from GENCODE hg38 protein FASTA
  (/home/data/genome/hg38_gencode_protein.fa)
- 99.8% coverage of training ENSG IDs found in GENCODE
- Sequences truncated to 512 amino acids (ESM2 max=1024, 512 is safe for memory)
- ESM2 precomputed at setup() time, then unloaded to free memory
- Zero vector fallback for genes not in GENCODE (< 2%)

ARCHITECTURE:
ESM2-650M (frozen, mean-pooled) → 1280-dim
  → LayerNorm(1280) + Linear(1280, 256) + GELU = esm2_feat [B, 256]
STRING_GNN (frozen buffer)
  → LookupEmbedding [B, 256] = str_feat
Gated Fusion:
  gate = sigmoid(Linear(512, 256)) where input=concat(esm2_feat, str_feat)
  fused = gate * esm2_feat + (1 - gate) * str_feat  → [B, 256]
Input Projection:
  LayerNorm(256) + Linear(256, 384) + GELU + Dropout(0.30) → [B, 384]
3 × PreNormResBlock(384, 768, dropout=0.30)
Output Head:
  LayerNorm(384) + Dropout(0.15) + Linear(384, 6640 × 3) → reshape [B, 3, 6640]
Per-gene bias: [6640, 3] (separate optimizer group, wd=0.05)

OPTIMIZER (3 parameter groups):
  Muon: 2D weight matrices in hidden blocks, lr=0.01, wd=0.01
  AdamW: all other params (projections, norms, gate, head), lr=3e-4, wd=0.01
  AdamW (gene_bias): per-gene bias, lr=3e-4, wd=0.05

TRAINING:
  Loss: Focal (gamma=2.0) with class weights + Manifold Mixup
  Mixup: prob=0.65, alpha=0.2 (higher prob proven for dual-branch vs STRING-only)
  Schedule: CosineAnnealingWarmRestarts T_0=80, T_mult=2 (cycles: 80, 160, 320)
  Patience: 80 (one full CosineWR cycle)
  Max epochs: 500
  Gradient clip: 2.0

TEST:
  Top-5 checkpoint ensemble (average raw logits before argmax)
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
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRING_GNN_DIR = "/home/Models/STRING_GNN"
GENCODE_PROTEIN_FASTA = "/home/data/genome/hg38_gencode_protein.fa"
ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ESM2_CACHE_DIR = "/home/.cache/huggingface/hub"
ESM2_MAX_SEQ_LEN = 512    # Truncation limit for ESM2 (safe for H100 memory)
ESM2_DIM = 1280            # ESM2-650M embedding dimension
N_GENES = 6640             # number of response genes per perturbation
N_CLASSES = 3              # down (-1→0), neutral (0→1), up (1→2)
GNN_DIM = 256              # STRING_GNN output dimension
HIDDEN_DIM = 384           # MLP hidden dimension (proven optimal for this task)
INNER_DIM = 768            # MLP inner (expansion) dimension


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Each sample is one gene perturbation experiment in HepG2 cells."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        if "label" in df.columns:
            # Labels in {-1,0,1} -> shift to {0,1,2}
            labels = np.array(
                [json.loads(x) for x in df["label"].tolist()], dtype=np.int64
            )
            self.labels: Optional[torch.Tensor] = torch.tensor(
                labels + 1, dtype=torch.long
            )
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
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
        micro_batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbDataset(train_df)
        self.val_ds = PerturbDataset(val_df)
        self.test_ds = PerturbDataset(test_df)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Model Components
# ---------------------------------------------------------------------------
class PreNormResBlock(nn.Module):
    """Pre-LayerNorm residual block (proven stable in node1-3-2 lineage).

    Architecture:
        output = x + LN(x) -> Linear(dim->inner) -> GELU -> Dropout
                               -> Linear(inner->dim) -> Dropout
    """

    def __init__(self, dim: int, inner_dim: int, dropout: float = 0.30) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Helpers: GENCODE protein FASTA → ENSG→sequence mapping
# ---------------------------------------------------------------------------
def _build_ensg_to_seq(fasta_path: str, max_seq_len: int = ESM2_MAX_SEQ_LEN) -> Dict[str, str]:
    """Parse GENCODE hg38 protein FASTA and build ENSG→longest-isoform-seq dict.

    FASTA header format:
      ENSP00000493376.2|ENST00000641515.2|ENSG00000186092.7|...|GENE_SYMBOL|length
    We extract the 3rd pipe-delimited field as the ENSG ID (version stripped).
    For genes with multiple isoforms, keep the longest sequence.
    Sequences are truncated to max_seq_len.
    """
    from Bio import SeqIO
    ensg_to_seq: Dict[str, str] = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        parts = record.id.split("|")
        if len(parts) >= 3:
            ensg_with_ver = parts[2]
            ensg = ensg_with_ver.split(".")[0]  # Strip version
            seq = str(record.seq)[:max_seq_len]
            # Keep longest isoform per ENSG
            if ensg not in ensg_to_seq or len(seq) > len(ensg_to_seq[ensg]):
                ensg_to_seq[ensg] = seq
    return ensg_to_seq


def _precompute_esm2_embeddings(
    ensg_ids: List[str],
    ensg_to_seq: Dict[str, str],
    device: torch.device,
    batch_size: int = 16,
) -> Dict[str, torch.Tensor]:
    """Precompute frozen ESM2-650M mean-pooled embeddings for a list of ENSG IDs.

    Returns a dict mapping ENSG_ID → float32 tensor of shape [1280].
    Genes not found in ensg_to_seq receive a zero vector.
    """
    from transformers import AutoTokenizer, EsmForMaskedLM

    print(f"Loading ESM2-650M tokenizer and model for precomputation...")
    tokenizer = AutoTokenizer.from_pretrained(
        ESM2_MODEL_NAME,
        cache_dir=ESM2_CACHE_DIR,
    )
    esm2_model = EsmForMaskedLM.from_pretrained(
        ESM2_MODEL_NAME,
        cache_dir=ESM2_CACHE_DIR,
        torch_dtype=torch.bfloat16,
    ).to(device)
    esm2_model.eval()

    # Special token IDs for mean-pool masking
    special_ids = torch.tensor(
        [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.eos_token_id],
        device=device,
    )

    embeddings: Dict[str, torch.Tensor] = {}

    # Separate genes with sequences from missing ones
    valid_genes = [g for g in ensg_ids if g in ensg_to_seq]
    missing_genes = [g for g in ensg_ids if g not in ensg_to_seq]

    print(f"  Computing ESM2 embeddings: {len(valid_genes)} found, {len(missing_genes)} missing (zero fallback)")

    # Process in batches
    for batch_start in range(0, len(valid_genes), batch_size):
        batch_genes = valid_genes[batch_start : batch_start + batch_size]
        batch_seqs = [ensg_to_seq[g] for g in batch_genes]

        # Tokenize
        encoded = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=ESM2_MAX_SEQ_LEN + 2,  # +2 for CLS/EOS tokens
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            output = esm2_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # Get last hidden state (ESM2-650M: layer 33)
        token_repr = output.hidden_states[-1]  # [B, L, 1280] bf16

        # Mean-pool over amino acid tokens (exclude CLS, EOS, PAD)
        mask = torch.isin(input_ids, special_ids)  # True for special tokens
        masked_repr = token_repr.masked_fill(mask.unsqueeze(-1), 0.0)
        count = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
        mean_repr = (masked_repr.sum(dim=1) / count).float()  # [B, 1280] in fp32

        for i, gene in enumerate(batch_genes):
            embeddings[gene] = mean_repr[i].cpu()

    # Assign zero vectors for missing genes
    zero_vec = torch.zeros(ESM2_DIM, dtype=torch.float32)
    for gene in missing_genes:
        embeddings[gene] = zero_vec.clone()

    # Free ESM2 memory immediately
    del esm2_model
    torch.cuda.empty_cache()
    print(f"  ESM2 precomputation done. Memory freed.")

    return embeddings


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        inner_dim: int = INNER_DIM,
        n_blocks: int = 3,
        dropout: float = 0.30,          # Trunk dropout
        head_dropout: float = 0.15,     # Output head dropout (proven optimal)
        muon_lr: float = 0.01,          # Muon LR
        adamw_lr: float = 3e-4,         # AdamW LR
        weight_decay: float = 0.01,     # MLP weight decay
        gene_bias_wd: float = 0.05,     # Per-gene bias dedicated weight decay
        label_smoothing: float = 0.0,   # No label smoothing for focal loss
        focal_gamma: float = 2.0,       # Focal loss gamma
        # CosineAnnealingWarmRestarts
        cosine_t0: int = 80,            # T_0=80 proven for dual-branch
        cosine_t_mult: int = 2,         # Cycle doubling
        cosine_eta_min: float = 1e-7,
        grad_clip_norm: float = 2.0,
        # Manifold Mixup
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.65,       # Higher prob proven for dual-branch
        # Checkpoint ensemble size
        top_k_ensemble: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Populated in setup()
        self.esm2_proj: Optional[nn.Sequential] = None   # ESM2 1280→256 projection
        self.gate_proj: Optional[nn.Linear] = None       # Gated fusion 512→256
        self.input_proj: Optional[nn.Sequential] = None  # 256→384 MLP input
        self.blocks: Optional[nn.ModuleList] = None      # PreNormResBlocks
        self.output_head: Optional[nn.Sequential] = None
        self.gene_bias: Optional[nn.Parameter] = None    # [N_GENES, N_CLASSES]

        # Lookup dicts
        self.gnn_id_to_idx: Dict[str, int] = {}
        self.esm2_id_to_idx: Dict[str, int] = {}  # ENSG → row in esm2_embeddings

        # Metric accumulators
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        """Build model, precompute frozen embeddings from ESM2 and STRING_GNN."""
        from transformers import AutoModel

        # ---- Step 1: Build GENCODE ENSG→protein_sequence mapping ----
        self.print("Building ENSG → protein sequence mapping from GENCODE FASTA...")
        ensg_to_seq = _build_ensg_to_seq(GENCODE_PROTEIN_FASTA, ESM2_MAX_SEQ_LEN)
        self.print(f"  GENCODE protein FASTA: {len(ensg_to_seq)} unique ENSG IDs")

        # ---- Step 2: Collect all unique ENSG IDs from train/val/test ----
        data_dir = Path(__file__).parent.parent.parent / "data"
        all_ensg_ids: set = set()
        for split in ["train", "val", "test"]:
            fpath = data_dir / f"{split}.tsv"
            if fpath.exists():
                df_split = pd.read_csv(str(fpath), sep="\t")
                all_ensg_ids.update(df_split["pert_id"].tolist())
        ensg_id_list = sorted(all_ensg_ids)
        self.print(f"  Total unique ENSG IDs across all splits: {len(ensg_id_list)}")

        # ---- Step 3: Precompute ESM2 embeddings ----
        # Each rank computes independently (small dataset, fast on H100)
        esm2_dict = _precompute_esm2_embeddings(
            ensg_ids=ensg_id_list,
            ensg_to_seq=ensg_to_seq,
            device=self.device,
            batch_size=16,
        )

        # Pack into a buffer [N_all_ensg, 1280]
        esm2_emb_matrix = torch.stack(
            [esm2_dict[g] for g in ensg_id_list], dim=0
        ).float()  # [N, 1280]
        self.register_buffer("esm2_embeddings", esm2_emb_matrix)
        self.esm2_id_to_idx = {g: i for i, g in enumerate(ensg_id_list)}
        n_covered = sum(1 for g in ensg_id_list if g in ensg_to_seq)
        self.print(
            f"  ESM2 buffer: {esm2_emb_matrix.shape}, "
            f"coverage: {n_covered}/{len(ensg_id_list)} ({100*n_covered/len(ensg_id_list):.1f}%)"
        )
        del esm2_dict, esm2_emb_matrix, ensg_to_seq
        torch.cuda.empty_cache()

        # ---- Step 4: Precompute STRING_GNN embeddings ----
        self.print("Loading STRING_GNN and computing frozen node embeddings...")
        gnn_model = AutoModel.from_pretrained(
            STRING_GNN_DIR, trust_remote_code=True
        )
        gnn_model.eval()
        gnn_model = gnn_model.to(self.device)

        graph = torch.load(
            Path(STRING_GNN_DIR) / "graph_data.pt",
            map_location=self.device,
        )
        edge_index = graph["edge_index"].to(self.device)
        edge_weight = graph.get("edge_weight")
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        with torch.no_grad():
            gnn_out = gnn_model(edge_index=edge_index, edge_weight=edge_weight)

        all_emb = gnn_out.last_hidden_state.detach().float()
        self.register_buffer("gnn_embeddings", all_emb)

        del gnn_model, gnn_out, graph, edge_index, edge_weight
        torch.cuda.empty_cache()

        node_names: List[str] = json.loads(
            (Path(STRING_GNN_DIR) / "node_names.json").read_text()
        )
        self.gnn_id_to_idx = {name: i for i, name in enumerate(node_names)}
        self.print(f"  STRING_GNN buffer: {all_emb.shape}")

        # ---- Step 5: Build model architecture ----
        hp = self.hparams

        # ESM2 projection branch: 1280 → 256
        self.esm2_proj = nn.Sequential(
            nn.LayerNorm(ESM2_DIM),
            nn.Linear(ESM2_DIM, GNN_DIM),  # 1280 → 256
            nn.GELU(),
        )

        # Gated sigmoidal fusion: concat(esm2_feat[256], str_feat[256]) → gate [256]
        # gate = sigmoid(Linear(512, 256)); fused = gate * esm2_feat + (1-gate) * str_feat
        self.gate_proj = nn.Linear(GNN_DIM * 2, GNN_DIM)  # 512 → 256

        # Input projection: 256 → 384
        self.input_proj = nn.Sequential(
            nn.LayerNorm(GNN_DIM),
            nn.Linear(GNN_DIM, hp.hidden_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # 3 PreNormResBlocks (proven optimal capacity)
        self.blocks = nn.ModuleList(
            [
                PreNormResBlock(hp.hidden_dim, hp.inner_dim, hp.dropout)
                for _ in range(hp.n_blocks)
            ]
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hp.hidden_dim),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.hidden_dim, N_GENES * N_CLASSES),
        )

        # Per-gene bias: [N_GENES, N_CLASSES] with dedicated high wd=0.05
        # Encodes population-level gene response tendencies (proven critical)
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES, N_CLASSES))

        # ---- Class weights (inverse frequency) ----
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq) / (1.0 / freq).mean()
        self.register_buffer("class_weights", class_weights)

        # Cast trainable params to float32 for stable optimization
        for k, v in self.named_parameters():
            if v.requires_grad:
                v.data = v.data.float()

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Architecture: ESM2-650M(1280→256) + STRING_GNN(256) -> "
            f"GatedFusion(256) -> Proj(384) -> "
            f"{hp.n_blocks}×PreNormResBlock({hp.hidden_dim},{hp.inner_dim}) "
            f"-> HeadDrop({hp.head_dropout}) -> Linear({hp.hidden_dim},{N_GENES}×{N_CLASSES}) "
            f"+ gene_bias[{N_GENES},{N_CLASSES}]"
        )
        self.print(f"Trainable params: {trainable:,} / {total:,}")
        self.print(f"FOCAL LOSS: gamma={hp.focal_gamma}")
        self.print(
            f"LR SCHEDULE: CosineAnnealingWarmRestarts "
            f"T_0={hp.cosine_t0}, T_mult={hp.cosine_t_mult}"
        )
        self.print(
            f"WD: MLP={hp.weight_decay}, gene_bias={hp.gene_bias_wd} (selective regularization)"
        )
        self.print(
            f"MANIFOLD MIXUP: alpha={hp.mixup_alpha}, prob={hp.mixup_prob} "
            f"(higher prob for dual-branch)"
        )

    # ------------------------------------------------------------------
    def _get_esm2_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Batch lookup of precomputed ESM2 embeddings."""
        indices = [
            self.esm2_id_to_idx.get(pid, -1) for pid in pert_ids
        ]
        result = []
        for idx in indices:
            if idx >= 0:
                result.append(self.esm2_embeddings[idx])
            else:
                result.append(torch.zeros(ESM2_DIM, device=self.device, dtype=torch.float32))
        return torch.stack(result, dim=0)  # [B, 1280]

    def _get_gnn_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Batch lookup of frozen STRING_GNN embeddings."""
        result = []
        for pid in pert_ids:
            row = self.gnn_id_to_idx.get(pid)
            if row is not None:
                result.append(self.gnn_embeddings[row])
            else:
                result.append(torch.zeros(GNN_DIM, device=self.device, dtype=torch.float32))
        return torch.stack(result, dim=0)  # [B, 256]

    def _gated_fusion(
        self, esm2_raw: torch.Tensor, str_emb: torch.Tensor
    ) -> torch.Tensor:
        """Gated sigmoidal fusion of ESM2 and STRING_GNN features.

        esm2_raw: [B, 1280]
        str_emb:  [B, 256]
        Returns:  [B, 256] fused representation
        """
        # Project ESM2: 1280 → 256
        esm2_feat = self.esm2_proj(esm2_raw)  # [B, 256]

        # Compute gate from concatenated features
        combined = torch.cat([esm2_feat, str_emb], dim=-1)  # [B, 512]
        gate = torch.sigmoid(self.gate_proj(combined))  # [B, 256]

        # Soft blend: gate controls how much ESM2 vs STRING contributes
        fused = gate * esm2_feat + (1.0 - gate) * str_emb  # [B, 256]
        return fused

    def _forward_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        """Forward pass from fused feature [B, 256] to logits [B, 3, N_GENES]."""
        x = self.input_proj(fused)  # [B, 384]
        for block in self.blocks:
            x = block(x)              # [B, 384]
        logits = self.output_head(x)                    # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)    # [B, 3, 6640]
        # Per-gene bias: [N_GENES, N_CLASSES] → [1, N_CLASSES, N_GENES]
        logits = logits + self.gene_bias.t().unsqueeze(0)
        return logits

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits [B, N_CLASSES, N_GENES]."""
        esm2_raw = self._get_esm2_emb(pert_ids)  # [B, 1280]
        str_emb = self._get_gnn_emb(pert_ids)    # [B, 256]
        fused = self._gated_fusion(esm2_raw, str_emb)  # [B, 256]
        return self._forward_from_fused(fused)

    def _manifold_mixup(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Manifold Mixup in the hidden MLP space (after input_proj, before residual blocks).

        Applied AFTER gated fusion and input_proj to avoid corrupting biological features.
        Higher prob=0.65 is proven effective for dual-branch architectures.
        """
        hp = self.hparams
        if self.training and np.random.random() < hp.mixup_prob:
            lam = np.random.beta(hp.mixup_alpha, hp.mixup_alpha)
            lam = max(lam, 1 - lam)  # Always take the larger weight
            batch_size = x.size(0)
            index = torch.randperm(batch_size, device=x.device)
            mixed_x = lam * x + (1 - lam) * x[index]
            return mixed_x, labels, labels[index], lam
        else:
            return x, labels, labels, 1.0

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        labels_b: Optional[torch.Tensor] = None,
        lam: float = 1.0,
    ) -> torch.Tensor:
        """Focal loss (gamma=2.0) with class weights and optional Manifold Mixup."""
        hp = self.hparams

        def _focal_loss_single(lgts, lbls):
            logits_flat = lgts.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
            labels_flat = lbls.reshape(-1)

            if hp.focal_gamma == 0.0:
                return F.cross_entropy(
                    logits_flat,
                    labels_flat,
                    weight=self.class_weights,
                    label_smoothing=hp.label_smoothing,
                )

            ce_per_sample = F.cross_entropy(
                logits_flat,
                labels_flat,
                weight=self.class_weights,
                reduction="none",
                label_smoothing=hp.label_smoothing,
            )
            with torch.no_grad():
                probs = F.softmax(logits_flat, dim=-1)
                pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
            focal_weight = (1.0 - pt.clamp(min=1e-8)) ** hp.focal_gamma
            return (focal_weight * ce_per_sample).mean()

        loss_a = _focal_loss_single(logits, labels)
        if labels_b is not None and lam < 1.0:
            loss_b = _focal_loss_single(logits, labels_b)
            return lam * loss_a + (1.0 - lam) * loss_b
        return loss_a

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pert_ids = batch["pert_id"]
        labels = batch["label"]

        # Dual-branch embedding lookup
        esm2_raw = self._get_esm2_emb(pert_ids)  # [B, 1280]
        str_emb = self._get_gnn_emb(pert_ids)    # [B, 256]
        fused = self._gated_fusion(esm2_raw, str_emb)  # [B, 256]

        # Input projection → hidden space for Manifold Mixup
        x = self.input_proj(fused)  # [B, 384]

        # Apply Manifold Mixup in the hidden representation space
        x, labels_a, labels_b, lam = self._manifold_mixup(x, labels)

        # Residual blocks
        for block in self.blocks:
            x = block(x)                               # [B, 384]
        logits = self.output_head(x)                   # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)   # [B, 3, 6640]
        logits = logits + self.gene_bias.t().unsqueeze(0)

        loss = self._compute_loss(logits, labels_a, labels_b, lam)
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        preds_local = torch.cat(self._val_preds, dim=0)
        labels_local = torch.cat(self._val_labels, dim=0)
        self._val_preds.clear()
        self._val_labels.clear()

        is_dist = dist.is_available() and dist.is_initialized()
        if is_dist:
            gathered_preds = self.all_gather(preds_local)
            gathered_labels = self.all_gather(labels_local)
            all_preds = gathered_preds.view(-1, N_CLASSES, N_GENES)
            all_labels = gathered_labels.view(-1, N_GENES)
            f1 = _compute_per_gene_f1(
                all_preds.cpu().numpy(), all_labels.cpu().numpy()
            )
            self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        else:
            f1 = _compute_per_gene_f1(preds_local.numpy(), labels_local.numpy())
            self.log("val_f1", f1, prog_bar=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        self._test_preds.append(logits.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        preds_local = torch.cat(self._test_preds, dim=0)
        self._test_preds.clear()

        gathered = self.all_gather(preds_local)
        all_preds = gathered.view(-1, N_CLASSES, N_GENES)

        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1
        local_pert_ids = list(self._test_pert_ids)
        local_symbols = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        gathered_pert_ids_list: List[List[str]] = [local_pert_ids]
        gathered_symbols_list: List[List[str]] = [local_symbols]
        if is_dist:
            obj_pids = [None] * world_size
            obj_syms = [None] * world_size
            dist.all_gather_object(obj_pids, local_pert_ids)
            dist.all_gather_object(obj_syms, local_symbols)
            gathered_pert_ids_list = obj_pids
            gathered_symbols_list = obj_syms

        if self.trainer.is_global_zero:
            all_pert_ids_flat = [pid for lst in gathered_pert_ids_list for pid in lst]
            all_symbols_flat = [sym for lst in gathered_symbols_list for sym in lst]

            # De-duplicate (DDP may replicate samples across ranks)
            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            preds_np = all_preds.cpu().numpy()
            for i, pid in enumerate(all_pert_ids_flat):
                if pid not in seen:
                    seen.add(pid)
                    dedup_ids.append(pid)
                    dedup_syms.append(all_symbols_flat[i])
                    dedup_preds.append(preds_np[i])

            self._current_test_preds = np.stack(dedup_preds, axis=0)
            self._current_test_ids = dedup_ids
            self._current_test_syms = dedup_syms

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        from muon import MuonWithAuxAdam

        hp = self.hparams

        # ---- Parameter groups ----
        # Group 1: Muon — 2D weight matrices in hidden residual blocks
        muon_params = [
            p for name, p in self.blocks.named_parameters()
            if p.ndim >= 2 and p.requires_grad
        ]
        muon_param_ids = set(id(p) for p in muon_params)

        # Group 3: gene_bias — dedicated high L2 penalty for memorization control
        gene_bias_params = [self.gene_bias]
        gene_bias_ids = {id(self.gene_bias)}

        # Group 2: AdamW — all other trainable params (ESM2 proj, gate, input_proj, head, norms)
        adamw_params = [
            p for p in self.parameters()
            if p.requires_grad
            and id(p) not in muon_param_ids
            and id(p) not in gene_bias_ids
        ]

        param_groups = [
            # Muon for hidden block weight matrices
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.weight_decay,
                momentum=0.95,
            ),
            # AdamW for all other params (projections, fusion, norms, head)
            dict(
                params=adamw_params,
                use_muon=False,
                lr=hp.adamw_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.weight_decay,
            ),
            # AdamW for gene_bias with dedicated high L2 penalty
            dict(
                params=gene_bias_params,
                use_muon=False,
                lr=hp.adamw_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.gene_bias_wd,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # CosineAnnealingWarmRestarts: T_0=80 proven for dual-branch architectures
        # Cycle lengths: 80 → 160 → 320 epochs (T_mult=2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hp.cosine_t0,
            T_mult=hp.cosine_t_mult,
            eta_min=hp.cosine_eta_min,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers: save only trainable params + buffers
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        trainable_state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_state_dict:
                    trainable_state_dict[key] = full_state_dict[key]

        for name, buffer in self.named_buffers():
            key = prefix + name
            if key in full_state_dict:
                trainable_state_dict[key] = full_state_dict[key]

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_buffers = sum(b.numel() for _, b in self.named_buffers())
        # Use plain print to avoid requiring Trainer context (e.g., during ensemble loading)
        print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%), plus {total_buffers} buffer values"
        )

        return trainable_state_dict

    def load_state_dict(self, state_dict, strict=True):
        full_state_keys = set(super().state_dict().keys())
        trainable_keys = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        buffer_keys = {
            name for name, _ in self.named_buffers() if name in full_state_keys
        }
        expected_keys = trainable_keys | buffer_keys

        missing_keys = [k for k in expected_keys if k not in state_dict]
        unexpected_keys = [k for k in state_dict if k not in expected_keys]

        # Use plain print to avoid requiring Trainer context (e.g., during ensemble loading)
        if missing_keys:
            print(f"Warning: Missing checkpoint keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

        loaded_trainable = len([k for k in state_dict if k in trainable_keys])
        loaded_buffers = len([k for k in state_dict if k in buffer_keys])
        print(
            f"Loading checkpoint: {loaded_trainable} trainable parameters and "
            f"{loaded_buffers} buffers"
        )

        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-averaged F1 exactly matching calc_metric.py."""
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [N_samples, N_genes]
    n_genes = labels.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels[:, g]
        yh = y_hat[:, g]
        per_class_f1 = sk_f1(
            yt, yh, labels=[0, 1, 2], average=None, zero_division=0
        )
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        f1_vals.append(float(per_class_f1[present].mean()))
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    """Save test predictions in TSV format required by calc_metric.py."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert len(pert_ids) == len(preds), (
        f"Length mismatch: {len(pert_ids)} pert_ids vs {len(preds)} pred rows"
    )
    rows = [
        {
            "idx": pert_ids[i],
            "input": symbols[i],
            "prediction": json.dumps(preds[i].tolist()),
        }
        for i in range(len(pert_ids))
    ]
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions -> {out_path}")


def _load_top_k_checkpoints(
    ckpt_dir: Path,
    top_k: int = 5,
) -> List[Path]:
    """Find and return the top-K checkpoints by val_f1 from the checkpoint directory."""
    import re
    ckpt_files = list(ckpt_dir.rglob("*.ckpt"))
    # Filter out 'last.ckpt'
    ckpt_files = [f for f in ckpt_files if "last" not in f.name]

    scored = []
    for f in ckpt_files:
        # Match val_f1=X.XXXX in filename (strip trailing dot before .ckpt extension)
        m = re.search(r"val_f1[=_]([\d.]+)", f.name)
        if m:
            val_str = m.group(1).rstrip(".")
            try:
                scored.append((float(val_str), f))
            except ValueError:
                pass

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in scored[:top_k]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Node1-2-2-3: Frozen ESM2-650M + STRING_GNN Dual-Branch + Gated Fusion "
            "+ 3-Block MLP (h=384) + Muon(lr=0.01) + FocalLoss(gamma=2.0) "
            "+ CosineWR(T_0=80) + ManifoldMixup(0.65) + TopK Ensemble"
        )
    )
    p.add_argument("--micro-batch-size", type=int, default=32,
                   help="Batch size per GPU (reduced for dual-branch memory)")
    p.add_argument("--global-batch-size", type=int, default=512,
                   help="Must be multiple of micro_batch_size * 8")
    p.add_argument("--max-epochs", type=int, default=500,
                   help="Provides 3+ CosineWR cycles (80+160+320)")
    p.add_argument("--muon-lr", type=float, default=0.01,
                   help="Muon optimizer LR for hidden block weights")
    p.add_argument("--adamw-lr", type=float, default=3e-4,
                   help="AdamW LR for other params")
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="Weight decay for MLP (Muon + AdamW)")
    p.add_argument("--gene-bias-wd", type=float, default=0.05,
                   help="Dedicated weight decay for per-gene bias (targeted L2 penalty)")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal loss gamma (2.0 proven optimal)")
    p.add_argument("--dropout", type=float, default=0.30,
                   help="Trunk dropout in MLP blocks")
    p.add_argument("--head-dropout", type=float, default=0.15,
                   help="Head dropout (proven optimal from tree history)")
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--inner-dim", type=int, default=768)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--cosine-t0", type=int, default=80,
                   help="CosineWR T_0=80 (longer cycles proven for dual-branch)")
    p.add_argument("--cosine-t-mult", type=int, default=2)
    p.add_argument("--cosine-eta-min", type=float, default=1e-7)
    p.add_argument("--grad-clip-norm", type=float, default=2.0)
    p.add_argument("--early-stop-patience", type=int, default=80,
                   help="One full CosineWR cycle (T_0=80) of patience")
    p.add_argument("--save-top-k", type=int, default=5,
                   help="Top-K checkpoints saved (for ensemble)")
    p.add_argument("--top-k-ensemble", type=int, default=5,
                   help="Number of checkpoints to ensemble at test time")
    p.add_argument("--mixup-alpha", type=float, default=0.2,
                   help="Manifold Mixup Beta distribution alpha")
    p.add_argument("--mixup-prob", type=float, default=0.65,
                   help="Mixup probability per batch (higher proven for dual-branch)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    p.add_argument("--val_check_interval", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # DataModule
    # ------------------------------------------------------------------
    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = PerturbModule(
        hidden_dim=args.hidden_dim,
        inner_dim=args.inner_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay,
        gene_bias_wd=args.gene_bias_wd,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        cosine_t0=args.cosine_t0,
        cosine_t_mult=args.cosine_t_mult,
        cosine_eta_min=args.cosine_eta_min,
        grad_clip_norm=args.grad_clip_norm,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        top_k_ensemble=args.top_k_ensemble,
    )

    # ------------------------------------------------------------------
    # Trainer configuration
    # ------------------------------------------------------------------
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = limit_val = limit_test = args.debug_max_step
        max_steps = args.debug_max_step
        val_check_interval = 1.0
        num_sanity_val_steps = 0
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1
        val_check_interval = args.val_check_interval
        num_sanity_val_steps = 2

    # Save top-5 checkpoints for ensemble at test time
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-val_f1={val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    # Patience=80: one full CosineWR cycle after best checkpoint
    # (proven: node3-3-1-2-1-1-1 peaked at epoch 137, stopped at 217 with patience=80)
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stop_patience,
        min_delta=1e-5,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    callbacks = [checkpoint_cb, early_stop_cb, lr_monitor, progress_bar]

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=300)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval if (args.debug_max_step is None and not args.fast_dev_run) else 1.0,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=callbacks,
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        gradient_clip_val=args.grad_clip_norm,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.fit(model, datamodule=datamodule)

    # ------------------------------------------------------------------
    # Test: Top-K checkpoint ensemble
    # ------------------------------------------------------------------
    if args.fast_dev_run or args.debug_max_step is not None:
        # Debug mode: use current model without ensemble
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        ckpt_dir = output_dir / "checkpoints"
        top_k_ckpts = _load_top_k_checkpoints(ckpt_dir, top_k=args.top_k_ensemble)

        if len(top_k_ckpts) < 2:
            # Fallback to best single checkpoint
            print(f"Found {len(top_k_ckpts)} checkpoints; using best single checkpoint.")
            test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")
        else:
            # Top-K ensemble: average raw logits from top-K checkpoints
            print(f"Running Top-{len(top_k_ckpts)} checkpoint ensemble...")
            all_ensemble_preds: Optional[np.ndarray] = None
            ensemble_ids: Optional[List[str]] = None
            ensemble_syms: Optional[List[str]] = None

            for ckpt_idx, ckpt_path in enumerate(top_k_ckpts):
                print(f"  Loading checkpoint {ckpt_idx+1}/{len(top_k_ckpts)}: {ckpt_path.name}")
                # Load checkpoint and run test
                test_model = PerturbModule.load_from_checkpoint(
                    str(ckpt_path),
                    map_location="cpu",
                )
                # Test with this checkpoint (single GPU to avoid DDP complications with ensembling)
                single_trainer = pl.Trainer(
                    accelerator="gpu",
                    devices=n_gpus,
                    num_nodes=1,
                    strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=300)),
                    precision="bf16-mixed",
                    logger=False,
                    enable_checkpointing=False,
                    enable_model_summary=False,
                )
                single_trainer.test(test_model, datamodule=datamodule)

                if single_trainer.is_global_zero:
                    ckpt_preds = test_model._current_test_preds   # [N, 3, 6640]
                    ckpt_ids = test_model._current_test_ids
                    ckpt_syms = test_model._current_test_syms

                    if all_ensemble_preds is None:
                        all_ensemble_preds = ckpt_preds.copy()
                        ensemble_ids = ckpt_ids
                        ensemble_syms = ckpt_syms
                    else:
                        # Average raw logits (averaging before argmax is better)
                        all_ensemble_preds += ckpt_preds

            if trainer.is_global_zero and all_ensemble_preds is not None:
                # Final averaged predictions
                final_preds = all_ensemble_preds / len(top_k_ckpts)
                out_path = output_dir / "test_predictions.tsv"
                _save_test_predictions(ensemble_ids, ensemble_syms, final_preds, out_path)
                print(f"Ensemble test predictions saved to {out_path}")

            test_results = {}

    # ------------------------------------------------------------------
    # Save test predictions (fallback path: from debug or single-ckpt test)
    # ------------------------------------------------------------------
    if trainer.is_global_zero:
        out_path = output_dir / "test_predictions.tsv"
        if not out_path.exists():
            # Predictions not yet saved (fallback case)
            if hasattr(model, "_current_test_preds"):
                _save_test_predictions(
                    model._current_test_ids,
                    model._current_test_syms,
                    model._current_test_preds,
                    out_path,
                )

    return test_results


if __name__ == "__main__":
    main()
