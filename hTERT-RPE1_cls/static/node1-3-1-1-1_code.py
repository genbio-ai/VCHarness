"""
Node 1-3-1-2: STRING_GNN + ESM2 Multi-Modal Fusion

Architecture:
  - STRING_GNN (FULLY FROZEN — proven optimal from node1-2)
    Precomputed frozen PPI embeddings: [N_nodes, 256]
  - ESM2-650M (FULLY FROZEN — new complementary signal)
    Precomputed frozen protein sequence embeddings: [N_genes, 1280]
    Protein sequences fetched from local GENCODE hg38 FASTA
    Mean-pooled across residues (excluding special tokens)
  - Feature fusion: concat([GNN_emb, ESM2_emb]) → [B, 1536]
  - MLP head reverted to node1-2 dimensions: hidden=512, rank=256
    (4 ResBlocks instead of 6, to control trainable param count)
  - Focal loss gamma=3.0 (increased from 2.0, per feedback)
  - Cosine annealing with warm restarts (T_0=50 epochs, T_mult=2)
  - Weight decay lowered to 5e-4 (per feedback)
  - Dropout=0.2 (same as node1-2, reduced from parent's 0.25)

Key changes from parent (node1-3-1-1, F1=0.4856):
  1. ADD ESM2-650M protein sequence embeddings [1280-dim] as complementary input
     - STRING_GNN captures PPI topology; ESM2 captures amino acid sequence features
     - Concatenated: [256+1280] = 1536-dim fused input
     - Both frozen: no backbone gradient, fast training
  2. REVERT head to node1-2 dimensions (hidden=512, rank=256)
     - Parent's 33M trainable params overfit on 1,416 samples
     - Reverted + smaller n_layers=4 → ~13M trainable params (better for data scale)
  3. INCREASE focal gamma: 2.0 → 3.0
     - Feedback from node1-3-1-1: increasing gamma to 3.0 expected +0.5–1.5% F1
     - Stronger focus on hard examples (minority up/down-regulated genes)
  4. COSINE ANNEALING WITH WARM RESTARTS (T_0=50 epochs, T_mult=2)
     - Replaces monotone cosine decay to escape local plateaus
     - Both prior nodes plateaued after epoch ~97; restarts at 50/150/350 explore new optima
  5. LOWER weight decay: 1e-3 → 5e-4
     - Calibrated for returned node1-2-scale head; parent used higher WD for larger model
  6. REDUCE n_residual_layers: 6 → 4
     - Controls parameter count given new 1536-dim input (vs 256-dim in prior nodes)
     - Total trainable params: ~13M vs ~33M parent, appropriate for 1,416 samples

Differentiation from siblings:
  - No sibling nodes exist for node1-3-1-1's children yet
  - This node explores: Multi-modal fusion (PPI + sequence) as first child

Expected performance: F1=0.49–0.53
  - Floor: parent's F1=0.4856 (STRING_GNN only; adding ESM2 should not hurt)
  - Main hypothesis: ESM2 captures amino acid sequence complementary to PPI topology,
    enabling the model to distinguish genes with similar network positions but different
    sequence-driven regulatory functions
  - Key risk: OOV genes for ESM2 (non-protein coding) may limit gains; mitigated by
    separate learned oov_esm2 embedding
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import json
import math
import argparse
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, EsmForMaskedLM

# ─── Constants ────────────────────────────────────────────────────────────────

N_GENES_OUT = 6640
N_CLASSES = 3
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ESM2_EMB_DIM = 1280
GNN_EMB_DIM = 256
PROTEIN_FASTA = Path("/home/data/genome/hg38_gencode_protein.fa")


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_logits_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Exact per-gene macro F1 matching calc_metric.py logic.

    Args:
        pred_logits_np: [N, 3, G] float (logits or probabilities)
        labels_np:      [N, G]    int   (class indices 0/1/2)

    Returns:
        Mean per-gene F1 score (float).
    """
    pred_classes = pred_logits_np.argmax(axis=1)  # [N, G]
    n_genes = labels_np.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = labels_np[:, g]
        yh = pred_classes[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── STRING_GNN Loading ───────────────────────────────────────────────────────

def load_frozen_string_gnn_embeddings(
    model_dir: Path,
    device: torch.device,
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """Load STRING_GNN model, extract frozen embeddings, then delete model."""
    node_names = json.loads((model_dir / "node_names.json").read_text())
    graph = torch.load(model_dir / "graph_data.pt", weights_only=False)
    node_name_to_idx = {name: i for i, name in enumerate(node_names)}

    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    edge_index = graph["edge_index"].to(device)
    edge_weight = graph.get("edge_weight")
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    with torch.no_grad():
        outputs = model(edge_index=edge_index, edge_weight=edge_weight)
    embeddings = outputs.last_hidden_state.float().cpu().numpy()

    del model, edge_index, edge_weight
    torch.cuda.empty_cache()

    return embeddings, node_names, node_name_to_idx


# ─── ESM2 Embedding Extraction ────────────────────────────────────────────────

def build_ensg_to_protein_seq(protein_fasta: Path) -> Dict[str, str]:
    """Build ENSG → longest protein sequence mapping from GENCODE protein FASTA.

    FASTA header format:
        >ENSP00000493376.2|ENST00000641515.2|ENSG00000186092.7|...|gene_symbol|length
    We extract the 3rd pipe-delimited field (ENSG ID with version) and strip the version.
    For genes with multiple isoforms, we keep the longest protein sequence.
    """
    ensg_to_seq: Dict[str, str] = {}

    if not protein_fasta.exists():
        print(f"[ESM2] WARNING: Protein FASTA not found at {protein_fasta}. ESM2 embeddings will be all OOV.")
        return ensg_to_seq

    try:
        from Bio import SeqIO
        with open(protein_fasta) as fh:
            for record in SeqIO.parse(fh, "fasta"):
                parts = record.id.split("|")
                if len(parts) < 3:
                    continue
                ensg_with_ver = parts[2]  # e.g. "ENSG00000186092.7"
                ensg = ensg_with_ver.split(".")[0]  # strip version → "ENSG00000186092"
                seq = str(record.seq)
                # Keep longest isoform per gene
                if ensg not in ensg_to_seq or len(seq) > len(ensg_to_seq[ensg]):
                    ensg_to_seq[ensg] = seq
    except Exception as e:
        print(f"[ESM2] WARNING: Failed to parse protein FASTA: {e}. ESM2 embeddings will be all OOV.")

    return ensg_to_seq


def compute_frozen_esm2_embeddings(
    all_pert_ids: List[str],
    ensg_to_seq: Dict[str, str],
    device: torch.device,
    batch_size: int = 32,
    max_aa_len: int = 1020,  # leave room for <cls> and <eos> tokens
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Precompute frozen ESM2-650M mean-pooled embeddings for all genes.

    Returns:
        embeddings: [N_with_protein, ESM2_EMB_DIM] float32 numpy array
        pert_to_idx: {ENSG_ID → row_index in embeddings} for genes with protein seqs
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Rank 0 downloads/caches the model first
    if local_rank == 0:
        _ = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
        _ = EsmForMaskedLM.from_pretrained(ESM2_MODEL_NAME, dtype=torch.float32)
    # Other ranks wait a moment to avoid concurrent download races
    else:
        import time
        time.sleep(10)

    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
    esm2 = EsmForMaskedLM.from_pretrained(ESM2_MODEL_NAME, dtype=torch.float32)
    esm2 = esm2.eval()
    for p in esm2.parameters():
        p.requires_grad = False
    esm2 = esm2.to(device)

    # Deduplicate pert_ids while preserving order
    seen: set = set()
    unique_ids: List[str] = []
    for pid in all_pert_ids:
        if pid not in seen:
            seen.add(pid)
            unique_ids.append(pid)

    # Only process genes that have protein sequences
    has_protein: List[str] = [pid for pid in unique_ids if pid in ensg_to_seq]
    print(f"[ESM2 rank {local_rank}] Genes with protein seqs: {len(has_protein)}/{len(unique_ids)}")

    # Pre-build special token tensor for masking
    special_ids_cpu = torch.tensor(
        [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.eos_token_id],
        dtype=torch.long,
    )

    all_embeddings: List[np.ndarray] = []
    all_emb_ids: List[str] = []

    for i in range(0, len(has_protein), batch_size):
        batch_ids = has_protein[i : i + batch_size]
        # Truncate to max_aa_len amino acids before tokenization
        seqs = [ensg_to_seq[pid][:max_aa_len] for pid in batch_ids]

        inputs = tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_aa_len + 2,  # +2 for <cls> and <eos>
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = esm2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Mean pool over residue tokens (exclude <cls>, <eos>, <pad>)
        hidden = outputs.hidden_states[-1]  # [B, seq_len, 1280]
        special_ids_dev = special_ids_cpu.to(device)
        is_special = torch.isin(input_ids, special_ids_dev)  # [B, seq_len]
        hidden_masked = hidden.masked_fill(is_special.unsqueeze(-1), 0.0)
        count = (~is_special).float().sum(dim=1, keepdim=True).clamp(min=1e-9)
        mean_emb = (hidden_masked.sum(dim=1) / count).float().cpu().numpy()  # [B, 1280]

        all_embeddings.append(mean_emb)
        all_emb_ids.extend(batch_ids)

        if (i // batch_size + 1) % 10 == 0:
            print(f"[ESM2 rank {local_rank}] Processed {i + len(batch_ids)}/{len(has_protein)} proteins")

    # Clean up ESM2 to free GPU memory
    del esm2, input_ids, attention_mask, hidden
    torch.cuda.empty_cache()

    if all_embeddings:
        embeddings_np = np.vstack(all_embeddings).astype(np.float32)
    else:
        embeddings_np = np.zeros((0, ESM2_EMB_DIM), dtype=np.float32)

    pert_to_idx: Dict[str, int] = {pid: i for i, pid in enumerate(all_emb_ids)}

    print(f"[ESM2 rank {local_rank}] Done. Embeddings shape: {embeddings_np.shape}")
    return embeddings_np, pert_to_idx


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset with STRING_GNN and ESM2 indices.

    Each sample provides:
    - pert_id:    Ensembl gene ID
    - symbol:     gene symbol
    - node_idx:   STRING_GNN node index (-1 if OOV)
    - esm2_idx:   ESM2 embedding index (-1 if no protein sequence)
    - label:      [N_GENES_OUT] integer class labels 0/1/2 (if has_labels)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        node_name_to_idx: Dict[str, int],
        esm2_pert_to_idx: Dict[str, int],
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        # STRING_GNN node indices
        self.node_indices = torch.tensor(
            [node_name_to_idx.get(pid, -1) for pid in self.pert_ids],
            dtype=torch.long,
        )

        # ESM2 embedding indices
        self.esm2_indices = torch.tensor(
            [esm2_pert_to_idx.get(pid, -1) for pid in self.pert_ids],
            dtype=torch.long,
        )

        self.has_labels = has_labels
        if has_labels and "label" in df.columns:
            rows = []
            for lbl_str in df["label"]:
                rows.append([x + 1 for x in json.loads(lbl_str)])
            self.labels = torch.tensor(rows, dtype=torch.long)
        else:
            self.has_labels = False

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int):
        item = {
            "pert_id":  self.pert_ids[idx],
            "symbol":   self.symbols[idx],
            "node_idx": self.node_indices[idx],
            "esm2_idx": self.esm2_indices[idx],
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule using precomputed frozen STRING_GNN + ESM2 embeddings."""

    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if hasattr(self, "train_ds"):
            return

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

        # ── STRING_GNN node name mapping ──────────────────────────────────────
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        self.node_name_to_idx = {name: i for i, name in enumerate(node_names)}
        self.n_nodes = len(node_names)

        # ── Load all splits ───────────────────────────────────────────────────
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        # ── ESM2 embedding precomputation ─────────────────────────────────────
        # Collect all unique pert_ids across all splits for precomputation
        all_pert_ids = (
            dfs["train"]["pert_id"].tolist()
            + dfs["val"]["pert_id"].tolist()
            + dfs["test"]["pert_id"].tolist()
        )

        print(f"[DataModule rank {local_rank}] Building ENSG→protein seq mapping...")
        ensg_to_seq = build_ensg_to_protein_seq(PROTEIN_FASTA)
        print(f"[DataModule rank {local_rank}] Proteins loaded: {len(ensg_to_seq)}")

        print(f"[DataModule rank {local_rank}] Computing frozen ESM2 embeddings...")
        self.esm2_emb_np, self.esm2_pert_to_idx = compute_frozen_esm2_embeddings(
            all_pert_ids, ensg_to_seq, device
        )

        # ── Coverage report ───────────────────────────────────────────────────
        train_gnn = sum(p in self.node_name_to_idx for p in dfs["train"]["pert_id"])
        train_esm2 = sum(p in self.esm2_pert_to_idx for p in dfs["train"]["pert_id"])
        print(
            f"[DataModule rank {local_rank}] Coverage — "
            f"STRING_GNN: {train_gnn}/{len(dfs['train'])} train genes, "
            f"ESM2: {train_esm2}/{len(dfs['train'])} train genes"
        )

        # ── Datasets ──────────────────────────────────────────────────────────
        self.train_ds = PerturbationDataset(
            dfs["train"], self.node_name_to_idx, self.esm2_pert_to_idx, True
        )
        self.val_ds = PerturbationDataset(
            dfs["val"], self.node_name_to_idx, self.esm2_pert_to_idx, True
        )
        self.test_ds = PerturbationDataset(
            dfs["test"], self.node_name_to_idx, self.esm2_pert_to_idx, True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )


# ─── Model ────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual MLP block: LayerNorm → Linear → GELU → Dropout → Linear → Dropout + skip."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MultiModalFrozenHead(nn.Module):
    """Prediction head combining frozen STRING_GNN + ESM2 embeddings via bilinear interaction.

    Key design:
    - Two frozen embedding buffers: GNN (256-dim) + ESM2 (1280-dim)
    - Separate OOV embeddings for each pathway
    - Concatenated to 1536-dim fused input
    - MLP head (hidden=512, 4 ResBlocks) — node1-2 scale, appropriate for data size
    - Bilinear interaction: rank=256 — proven effective in node1-2
    """

    def __init__(
        self,
        frozen_gnn_emb: torch.Tensor,      # [N_gnn_nodes, 256]
        frozen_esm2_emb: torch.Tensor,     # [N_esm2_genes, 1280]
        gnn_dim: int = 256,
        esm2_dim: int = 1280,
        hidden_dim: int = 512,
        bilinear_rank: int = 256,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        dropout: float = 0.2,
        n_residual_layers: int = 4,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_genes_out = n_genes_out
        self.bilinear_rank = bilinear_rank
        self.gnn_dim = gnn_dim
        self.esm2_dim = esm2_dim
        fused_dim = gnn_dim + esm2_dim  # 256 + 1280 = 1536

        # Frozen embedding buffers (not trainable parameters)
        self.register_buffer("frozen_gnn_emb", frozen_gnn_emb.float())
        if frozen_esm2_emb.shape[0] > 0:
            self.register_buffer("frozen_esm2_emb", frozen_esm2_emb.float())
        else:
            # Edge case: no proteins in ESM2
            self.register_buffer("frozen_esm2_emb", torch.zeros(1, esm2_dim))

        # OOV fallback embeddings — learned, one vector each
        self.oov_gnn_emb  = nn.Embedding(1, gnn_dim)
        self.oov_esm2_emb = nn.Embedding(1, esm2_dim)

        # Input normalization on fused representation
        self.input_norm = nn.LayerNorm(fused_dim)

        # Projection: fused_dim → hidden_dim
        self.proj_in = nn.Linear(fused_dim, hidden_dim)

        # Residual MLP head — node1-2 scale (4 blocks)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, expand=4, dropout=dropout)
             for _ in range(n_residual_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear projection: hidden_dim → n_classes * bilinear_rank
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * bilinear_rank)
        self.head_dropout = nn.Dropout(dropout)

        # Output gene embeddings [n_genes_out, bilinear_rank] — random init
        self.out_gene_emb = nn.Parameter(torch.empty(n_genes_out, bilinear_rank))
        nn.init.normal_(self.out_gene_emb, std=0.02)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.oov_gnn_emb.weight,  std=0.02)
        nn.init.normal_(self.oov_esm2_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)

    def forward(
        self,
        node_idx: torch.Tensor,   # [B] long, -1 for GNN OOV
        esm2_idx: torch.Tensor,   # [B] long, -1 for ESM2 OOV (no protein)
    ) -> torch.Tensor:
        """
        Returns:
            logits: [B, 3, 6640]
        """
        B = node_idx.shape[0]
        device = node_idx.device
        zeros = torch.zeros(B, dtype=torch.long, device=device)

        # ── Path 1: STRING_GNN frozen embedding ──────────────────────────────
        in_gnn_mask = (node_idx >= 0)
        gnn_emb = self.frozen_gnn_emb[node_idx.clamp(min=0)]        # [B, 256]
        oov_gnn = self.oov_gnn_emb(zeros)                            # [B, 256]
        in_gnn_f = in_gnn_mask.float().unsqueeze(1)                  # [B, 1]
        gnn_emb = gnn_emb * in_gnn_f + oov_gnn * (1.0 - in_gnn_f)  # [B, 256]

        # ── Path 2: ESM2 frozen embedding ─────────────────────────────────────
        in_esm2_mask = (esm2_idx >= 0)
        esm2_emb = self.frozen_esm2_emb[esm2_idx.clamp(min=0)]        # [B, 1280]
        oov_esm2 = self.oov_esm2_emb(zeros)                            # [B, 1280]
        in_esm2_f = in_esm2_mask.float().unsqueeze(1)                  # [B, 1]
        esm2_emb = esm2_emb * in_esm2_f + oov_esm2 * (1.0 - in_esm2_f)  # [B, 1280]

        # ── Fusion ────────────────────────────────────────────────────────────
        x = torch.cat([gnn_emb, esm2_emb], dim=1)  # [B, 1536]

        # ── MLP head ──────────────────────────────────────────────────────────
        x = self.input_norm(x)
        x = self.proj_in(x)           # [B, hidden_dim]
        for blk in self.res_blocks:
            x = blk(x)
        x = self.norm_out(x)          # [B, hidden_dim]

        # ── Bilinear interaction ───────────────────────────────────────────────
        x = self.head_dropout(x)
        pert_proj = self.proj_bilinear(x)                               # [B, n_classes * rank]
        pert_proj = pert_proj.view(B, self.n_classes, self.bilinear_rank)  # [B, 3, rank]

        # logits: [B, 3, rank] × [n_genes_out, rank]^T → [B, 3, n_genes_out]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)  # [B, 3, 6640]
        return logits


# ─── Focal Loss ───────────────────────────────────────────────────────────────

def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 3.0,
) -> torch.Tensor:
    """Focal loss: -(1-p_t)^gamma * log(p_t).
    gamma=3.0 (increased from 2.0) to focus more aggressively on hard examples.
    No label smoothing — node1-2-2 demonstrated it hurts F1.
    """
    B, C, G = logits.shape
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
    labels_flat = labels.reshape(-1)                        # [B*G]

    ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction="none")  # [B*G]

    with torch.no_grad():
        probs = F.softmax(logits_flat, dim=1)
        pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt).pow(gamma)

    return (focal_weight * ce_loss).mean()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(
    local_preds: torch.Tensor,
    local_labels: torch.Tensor,
    device: torch.device,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather variable-length tensors from all DDP ranks with padding."""
    local_size = torch.tensor([local_preds.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = int(max(s.item() for s in all_sizes))

    pad = max_size - local_preds.shape[0]
    p = local_preds.to(device)
    l = local_labels.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], dim=0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], dim=0)

    g_preds  = [torch.zeros_like(p) for _ in range(world_size)]
    g_labels = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(g_preds,  p)
    dist.all_gather(g_labels, l)

    real_preds  = torch.cat([g_preds[i][: all_sizes[i].item()].cpu()  for i in range(world_size)], 0)
    real_labels = torch.cat([g_labels[i][: all_sizes[i].item()].cpu() for i in range(world_size)], 0)
    return real_preds, real_labels


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbationLitModule(pl.LightningModule):
    """LightningModule for gene-perturbation DEG prediction (Node 1-3-1-2).

    Key design: frozen STRING_GNN + ESM2 multi-modal fusion → bilinear head.
    No backbone forward pass during training — only embedding table lookups.
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        esm2_dim: int = 1280,
        hidden_dim: int = 512,
        bilinear_rank: int = 256,
        n_residual_layers: int = 4,
        dropout: float = 0.2,
        lr: float = 5e-4,
        weight_decay: float = 5e-4,
        focal_gamma: float = 3.0,
        warmup_steps: int = 50,
        lr_T0_epochs: int = 50,       # CosineAnnealingWarmRestarts T_0 in epochs
        lr_T_mult: int = 2,            # CosineAnnealingWarmRestarts T_mult
        steps_per_epoch: int = 22,     # filled in main() after data is loaded
        n_nodes: int = 18870,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:      List[torch.Tensor] = []
        self._val_labels:     List[torch.Tensor] = []
        self._test_preds:     List[torch.Tensor] = []
        self._test_labels:    List[torch.Tensor] = []
        self._test_pert_ids:  List[str] = []
        self._test_symbols:   List[str] = []

    def setup(self, stage: Optional[str] = None):
        if hasattr(self, "model"):
            return

        hp = self.hparams
        device = self.device if self.device.type != "meta" else torch.device("cpu")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # ── Load STRING_GNN frozen embeddings ─────────────────────────────────
        if local_rank == 0:
            print(f"[Node1-3-1-2] Loading STRING_GNN embeddings on device={device}")
        frozen_gnn_np, _, _ = load_frozen_string_gnn_embeddings(STRING_GNN_DIR, device)
        frozen_gnn_tensor = torch.from_numpy(frozen_gnn_np).float()

        # ── Load ESM2 frozen embeddings from DataModule cache ─────────────────
        dm = self.trainer.datamodule
        if dm is not None and hasattr(dm, "esm2_emb_np"):
            esm2_emb_np = dm.esm2_emb_np
        else:
            # Fallback: empty ESM2 (all OOV); model degrades gracefully
            print(f"[Node1-3-1-2] WARNING: DataModule ESM2 embeddings not found, using empty buffer")
            esm2_emb_np = np.zeros((0, ESM2_EMB_DIM), dtype=np.float32)

        frozen_esm2_tensor = torch.from_numpy(esm2_emb_np).float()
        if local_rank == 0:
            print(
                f"[Node1-3-1-2] GNN emb: {frozen_gnn_tensor.shape}, "
                f"ESM2 emb: {frozen_esm2_tensor.shape}"
            )

        # ── Build model ───────────────────────────────────────────────────────
        self.model = MultiModalFrozenHead(
            frozen_gnn_emb=frozen_gnn_tensor,
            frozen_esm2_emb=frozen_esm2_tensor,
            gnn_dim=hp.gnn_dim,
            esm2_dim=hp.esm2_dim,
            hidden_dim=hp.hidden_dim,
            bilinear_rank=hp.bilinear_rank,
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            dropout=hp.dropout,
            n_residual_layers=hp.n_residual_layers,
        )

        # Cast all trainable parameters to float32
        for _, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

    def forward(self, node_idx: torch.Tensor, esm2_idx: torch.Tensor) -> torch.Tensor:
        return self.model(node_idx, esm2_idx)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return focal_loss(logits, labels, gamma=self.hparams.focal_gamma)

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_idx"], batch["esm2_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["node_idx"], batch["esm2_idx"])
        if "label" in batch:
            loss = self._compute_loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())
        return logits

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        local_p = torch.cat(self._val_preds,  dim=0)
        local_l = torch.cat(self._val_labels, dim=0)

        if self.trainer.world_size > 1:
            all_p, all_l = _gather_tensors(local_p, local_l, self.device, self.trainer.world_size)
        else:
            all_p, all_l = local_p, local_l

        f1 = compute_per_gene_f1(all_p.numpy(), all_l.numpy())
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["node_idx"], batch["esm2_idx"])
        probs = torch.softmax(logits, dim=1)  # [B, 3, 6640]
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs = torch.cat(self._test_preds, dim=0)
        dummy_labels = torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        if self._test_labels:
            dummy_labels = torch.cat(self._test_labels, dim=0)

        if self.trainer.world_size > 1:
            all_probs, all_labels = _gather_tensors(
                local_probs, dummy_labels, self.device, self.trainer.world_size
            )
            all_pert = [None] * self.trainer.world_size
            all_syms = [None] * self.trainer.world_size
            dist.all_gather_object(all_pert, self._test_pert_ids)
            dist.all_gather_object(all_syms, self._test_symbols)
            all_pert = [p for sub in all_pert for p in sub]
            all_syms = [s for sub in all_syms for s in sub]
        else:
            all_probs  = local_probs
            all_labels = dummy_labels
            all_pert   = self._test_pert_ids
            all_syms   = self._test_symbols

        if self.trainer.is_global_zero:
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"
            seen_ids: set = set()
            dedup_probs: list = []
            dedup_labels: list = []
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for i, (pert_id, symbol, probs) in enumerate(
                    zip(all_pert, all_syms, all_probs.numpy())
                ):
                    if pert_id not in seen_ids:
                        seen_ids.add(pert_id)
                        fh.write(f"{pert_id}\t{symbol}\t{json.dumps(probs.tolist())}\n")
                        dedup_probs.append(probs)
                        dedup_labels.append(all_labels[i].numpy())
            self.print(
                f"[Node1-3-1-2] Saved test predictions → {pred_path} ({len(seen_ids)} unique samples)"
            )

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-3-1-2] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=hp.lr,
            weight_decay=hp.weight_decay,
        )

        # Cosine annealing with warm restarts + linear warmup
        # T_0: restart period in steps (50 epochs × steps_per_epoch)
        # T_mult: doubles the restart period after each restart
        # Warmup: linear ramp for first warmup_steps steps
        T0_steps = hp.lr_T0_epochs * hp.steps_per_epoch
        T_mult = hp.lr_T_mult
        warmup = hp.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                # Linear warmup
                return float(step) / max(1, warmup)

            t = step - warmup
            # Compute current position within restart cycle
            T_cur_steps = T0_steps
            t_remaining = t
            while t_remaining >= T_cur_steps:
                t_remaining -= T_cur_steps
                T_cur_steps = int(T_cur_steps * T_mult)

            progress = t_remaining / max(1, T_cur_steps)
            # Cosine decay within cycle: 1.0 → 0.01
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable params ──────────────────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_sd = {}
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        for k, v in full_sd.items():
            if k in trainable_keys or k in buffer_keys:
                trainable_sd[k] = v
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving ckpt: {trained}/{total} trainable params ({100*trained/total:.1f}%)"
        )
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 1-3-1-2 — STRING_GNN + ESM2 Multi-Modal Fusion"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--gnn-dim",            type=int,   default=256)
    p.add_argument("--esm2-dim",           type=int,   default=1280)
    p.add_argument("--hidden-dim",         type=int,   default=512)
    p.add_argument("--bilinear-rank",      type=int,   default=256)
    p.add_argument("--n-residual-layers",  type=int,   default=4)
    p.add_argument("--dropout",            type=float, default=0.2)
    p.add_argument("--lr",                 type=float, default=5e-4)
    p.add_argument("--weight-decay",       type=float, default=5e-4)
    p.add_argument("--focal-gamma",        type=float, default=3.0)
    p.add_argument("--warmup-steps",       type=int,   default=50)
    p.add_argument("--lr-t0-epochs",       type=int,   default=50)
    p.add_argument("--lr-t-mult",          type=int,   default=2)
    p.add_argument("--micro-batch-size",   type=int,   default=16)
    p.add_argument("--global-batch-size",  type=int,   default=64)
    p.add_argument("--max-epochs",         type=int,   default=300)
    p.add_argument("--patience",           type=int,   default=50)
    p.add_argument("--num-workers",        type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",     type=int,   default=None)
    p.add_argument("--fast-dev-run",       action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── DataModule setup ──────────────────────────────────────────────────────
    dm = PerturbationDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()  # Precomputes ESM2 embeddings and builds datasets

    # ── Estimate steps per epoch for LR schedule ──────────────────────────────
    steps_per_epoch_raw = max(1, len(dm.train_ds) // (args.micro_batch_size * n_gpus))
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch_raw // accum)

    # ── LightningModule ───────────────────────────────────────────────────────
    lit = PerturbationLitModule(
        gnn_dim=args.gnn_dim,
        esm2_dim=args.esm2_dim,
        hidden_dim=args.hidden_dim,
        bilinear_rank=args.bilinear_rank,
        n_residual_layers=args.n_residual_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        warmup_steps=args.warmup_steps,
        lr_T0_epochs=args.lr_t0_epochs,
        lr_T_mult=args.lr_t_mult,
        steps_per_epoch=effective_steps_per_epoch,
        n_nodes=dm.n_nodes,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-5)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    # ── Debug settings ────────────────────────────────────────────────────────
    max_steps:           int         = -1
    limit_train_batches: float | int = 1.0
    limit_val_batches:   float | int = 1.0
    limit_test_batches:  float | int = 1.0
    fast_dev_run = False

    if args.debug_max_step is not None:
        max_steps           = args.debug_max_step
        limit_train_batches = args.debug_max_step
        limit_val_batches   = 2
        limit_test_batches  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accum,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)

    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    test_results = trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            "Node 1-3-1-2 — STRING_GNN + ESM2 Multi-Modal Fusion\n"
            f"Test results from trainer: {test_results}\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
