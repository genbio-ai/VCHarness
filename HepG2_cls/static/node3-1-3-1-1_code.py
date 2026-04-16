"""node3-1-3-1-1: ESM2-3B (frozen) + STRING_GNN (frozen) Dual-Branch
               + FUSION_DIM=512 Gated Fusion + 3-Block Pre-Norm MLP (h=384)
               + Muon+AdamW + Manifold Mixup (prob=0.75, soft-label KL)
               + CosineAnnealingWarmRestarts + Filtered Top-3 Ensemble

Parent: node3-1-3-1 (ESM2-150M frozen + STRING_GNN + gated fusion, test F1=0.5136)

Key changes from parent node3-1-3-1:
-----------------------------------
1. UPGRADE TO ESM2-3B: Replace frozen ESM2-150M (640-dim) with frozen ESM2-3B (2560-dim).
   The tree-best node3-1-1-1-1-2-1-1-1 (F1=0.5283) uses ESM2-3B + FUSION_DIM=512.
   Critical lesson: maintaining a 5:1 compression ratio (2560→512) via two-stage projection
   avoids the 10:1 bottleneck that prevented gains in node3-1-1-1-1-2-1
   (ESM2-3B + FUSION_DIM=256 → F1=0.5243, same as ESM2-650M, no improvement).

2. TWO-LAYER ESM2 PROJECTION: LN(2560) → Linear(2560→1024) → GELU → Linear(1024→512)
   instead of LN(640) → Linear(640→256). This graduated compression preserves the richer
   ESM2-3B 2560-dim representations before the gated fusion bottleneck. Two-stage projection
   provides ~2.5M additional projection capacity to handle the higher ESM2 dimensionality.

3. WIDER FUSION: FUSION_DIM=512 (from 256, +256 units). Proven in tree:
   node3-1-1-1-1-2-1-1 (ESM2-3B + FUSION_DIM=512 → F1=0.5265, +0.0022 over FUSION_DIM=256).
   STRING branch also updated to Linear(256→512) to match the wider fusion dimension.
   GatedFusion gate_proj now Linear(1024→512) for the larger concatenated input.

4. INCREASED MIXUP PROBABILITY: 0.65 → 0.75. The tree-best node3-1-1-1-1-2-1-1-1
   (F1=0.5283) uses mixup_prob=0.75. Stronger Manifold Mixup regularization is needed
   for the richer dual-modality representations to prevent overfitting on 1,273 samples.

5. FILTERED TOP-3 ENSEMBLE: Top-3 with ±0.003 F1 threshold filter instead of simple
   Top-5 average. Tree evidence: node3-1-1-1-1-2-1-1-1 showed that top-3 filtered
   (F1=0.5283) outperforms top-7 simple average (F1=0.5265, -0.0018). Simple averaging
   in parent gave only +0.0003 because top-5 checkpoints had similar quality (range
   0.5082–0.5133). Threshold filtering focuses ensemble on high-quality checkpoints.

Memory sources:
  - node3-1-3-1 feedback: ESM2-650M recommended; Parent F1=0.5136
  - node3-1-1-1-1-2-1-1-1 (F1=0.5283, tree best): ESM2-3B + FUSION_DIM=512 + top-3 filtered
  - node3-1-1-1-1-2-1-1 (F1=0.5265): FUSION_DIM=512 + ESM2-3B → proven +0.0022
  - node3-1-1-1-1-2-1 (F1=0.5243): ESM2-3B with 10:1 compression = bottleneck
  - node3-1-1-1-1-2 (F1=0.5243): ESM2-650M + FUSION_DIM=256, same as ESM2-3B bottleneck
  - node3-3-1-2-1-1-1 (F1=0.5243): same architecture, confirmed frozen ESM2-3B best recipe

Protein sequences sourced from: /home/data/genome/hg38_gencode_protein.fa
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

try:
    from muon import MuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    print("WARNING: muon package not found, falling back to AdamW only", flush=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRING_GNN_DIR = "/home/Models/STRING_GNN"
PROTEIN_FASTA = "/home/data/genome/hg38_gencode_protein.fa"
ESM2_MODEL_NAME = "facebook/esm2_t36_3B_UR50D"   # Upgraded: ESM2-3B (2560-dim)

ESM2_DIM = 2560       # ESM2-3B hidden dimension (was 640 for ESM2-150M)
GNN_DIM = 256         # STRING_GNN output embedding dimension
FUSION_DIM = 512      # Gated fusion output dimension (was 256, now 512 for richer fusion)
N_GENES = 6640        # Number of response genes per perturbation
N_CLASSES = 3         # down (-1→0), neutral (0→1), up (1→2)

# Fallback protein sequence for genes not found in FASTA
FALLBACK_SEQ = "MAAAAA"


# ---------------------------------------------------------------------------
# Protein sequence loading from FASTA
# ---------------------------------------------------------------------------
def _build_ensg_to_seq(fasta_path: str) -> Dict[str, str]:
    """Build ENSG → longest protein sequence map from GENCODE protein FASTA."""
    ensg2seqs: Dict[str, List[str]] = {}
    current_ensg: Optional[str] = None
    current_seq_parts: List[str] = []

    def flush():
        if current_ensg is not None and current_seq_parts:
            seq = "".join(current_seq_parts)
            if current_ensg not in ensg2seqs:
                ensg2seqs[current_ensg] = []
            ensg2seqs[current_ensg].append(seq)

    with open(fasta_path, "r") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                flush()
                current_seq_parts = []
                current_ensg = None
                for part in line[1:].split():
                    if part.startswith("gene:ENSG"):
                        current_ensg = part.split(":")[1]
                        break
            else:
                current_seq_parts.append(line)
    flush()

    # Keep the longest isoform per gene
    return {ensg: max(seqs, key=len) for ensg, seqs in ensg2seqs.items()}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Each sample is one gene perturbation experiment in HepG2 cells."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        if "label" in df.columns:
            # Labels in {-1, 0, 1} → shift to {0, 1, 2}
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
    """Pre-LayerNorm residual MLP block.

    output = x + dropout(fc2(dropout(gelu(fc1(ln(x))))))
    """

    def __init__(self, dim: int, inner_dim: int, dropout: float = 0.30) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class GatedFusion(nn.Module):
    """Learnable gated fusion of STRING and ESM2 embedding branches.

    gate = sigmoid(Linear(concat([h_str, h_esm]) → fusion_dim))
    fused = gate * h_str + (1 - gate) * h_esm
    """

    def __init__(self, in_dim: int = 512, fusion_dim: int = 512) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(in_dim * 2, fusion_dim)

    def forward(self, h_str: torch.Tensor, h_esm: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(torch.cat([h_str, h_esm], dim=-1)))
        return gate * h_str + (1.0 - gate) * h_esm


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = 384,
        inner_dim: int = 768,
        n_blocks: int = 3,
        dropout: float = 0.30,
        head_dropout: float = 0.10,
        muon_lr: float = 0.01,
        adamw_lr: float = 3e-4,
        weight_decay: float = 8e-4,
        cosine_t0: int = 80,
        cosine_t_mult: int = 2,
        min_lr: float = 1e-7,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.75,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Populated in setup()
        self.str_proj: Optional[nn.Sequential] = None
        self.esm_proj: Optional[nn.Sequential] = None
        self.gate_fusion: Optional[GatedFusion] = None
        self.input_proj: Optional[nn.Sequential] = None
        self.blocks: Optional[nn.ModuleList] = None
        self.output_head: Optional[nn.Sequential] = None
        self.gene_bias: Optional[nn.Parameter] = None

        # ENSG-ID → embedding index mappings
        self.gnn_id_to_idx: Dict[str, int] = {}
        self.esm2_id_to_idx: Dict[str, int] = {}

        # Metric accumulators
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        """Build model and precompute frozen embeddings from both branches."""
        from transformers import AutoModel, AutoTokenizer, EsmForMaskedLM

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # ---------------------------------------------------------------- #
        # STRING_GNN frozen embeddings                                      #
        # ---------------------------------------------------------------- #
        self.print("Loading STRING_GNN for frozen PPI embeddings ...")
        gnn_model_path = Path(STRING_GNN_DIR)
        gnn_model = AutoModel.from_pretrained(str(gnn_model_path), trust_remote_code=True)
        gnn_model.eval().to(self.device)

        graph = torch.load(gnn_model_path / "graph_data.pt", map_location=self.device)
        edge_index = graph["edge_index"].to(self.device)
        edge_weight = graph.get("edge_weight")
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        with torch.no_grad():
            gnn_out = gnn_model(edge_index=edge_index, edge_weight=edge_weight)

        gnn_emb = gnn_out.last_hidden_state.detach().float()
        self.register_buffer("gnn_embeddings", gnn_emb)  # [N_gnn, 256]

        del gnn_model, gnn_out, graph, edge_index
        if edge_weight is not None:
            del edge_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        node_names: List[str] = json.loads(
            (gnn_model_path / "node_names.json").read_text()
        )
        self.gnn_id_to_idx = {name: i for i, name in enumerate(node_names)}
        self.print(f"STRING_GNN: {gnn_emb.shape}, {len(self.gnn_id_to_idx)} gene IDs")

        # ---------------------------------------------------------------- #
        # ESM2-3B frozen embeddings (upgraded from ESM2-150M)              #
        # ---------------------------------------------------------------- #
        self.print(f"Loading {ESM2_MODEL_NAME} for frozen protein embeddings ...")

        # Rank 0 downloads first, then barrier
        if local_rank == 0:
            AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        esm2_tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)

        # Build ENSG → protein sequence map
        self.print(f"Parsing protein FASTA: {PROTEIN_FASTA} ...")
        ensg2seq = _build_ensg_to_seq(PROTEIN_FASTA)
        self.print(f"FASTA: {len(ensg2seq)} ENSG → protein sequences")

        # Collect all unique ENSG IDs from datamodule
        all_ensg_ids: List[str] = []
        if (hasattr(self, "trainer") and self.trainer is not None
                and self.trainer.datamodule is not None):
            dm = self.trainer.datamodule
            for split_attr in ["train_ds", "val_ds", "test_ds"]:
                split_ds = getattr(dm, split_attr, None)
                if split_ds is not None:
                    all_ensg_ids.extend(split_ds.pert_ids)
        unique_ensg_ids = list(dict.fromkeys(all_ensg_ids))
        self.print(f"Unique ENSG IDs for ESM2 embedding: {len(unique_ensg_ids)}")

        # Load ESM2-3B (bfloat16 for memory efficiency)
        if local_rank == 0:
            EsmForMaskedLM.from_pretrained(ESM2_MODEL_NAME)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        esm2_model = EsmForMaskedLM.from_pretrained(
            ESM2_MODEL_NAME, dtype=torch.bfloat16
        )
        esm2_model.eval().to(self.device)

        # Precompute ESM2-3B embeddings in batches
        # Reduced batch_size_esm=8 for ESM2-3B's larger memory footprint
        self.print(f"Precomputing {ESM2_MODEL_NAME} embeddings for {len(unique_ensg_ids)} genes ...")
        esm2_emb_list: List[torch.Tensor] = []
        batch_size_esm = 8  # Reduced from 32: ESM2-3B (2560-dim) needs more GPU memory per batch

        for start in range(0, len(unique_ensg_ids), batch_size_esm):
            batch_ids = unique_ensg_ids[start: start + batch_size_esm]
            # Use longest isoform, truncate to 1022 (leave room for <cls>, <eos>)
            batch_seqs = [ensg2seq.get(eid, FALLBACK_SEQ)[:1022] for eid in batch_ids]

            tokenized = esm2_tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            with torch.no_grad():
                outputs = esm2_model(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    output_hidden_states=True,
                )

            # Mean-pool over non-special tokens (last hidden state [B, seq_len, 2560])
            hidden = outputs["hidden_states"][-1]  # [B, seq_len, 2560]
            special_ids = torch.tensor(
                [esm2_tokenizer.pad_token_id,
                 esm2_tokenizer.cls_token_id,
                 esm2_tokenizer.eos_token_id],
                device=self.device,
            )
            special_mask = torch.isin(tokenized["input_ids"], special_ids)  # [B, seq]
            masked = hidden.masked_fill(special_mask.unsqueeze(-1), 0.0)
            counts = (~special_mask).float().sum(dim=1, keepdim=True).clamp(min=1e-9)
            mean_emb = (masked.sum(dim=1) / counts).float().cpu()  # [B, 2560]
            esm2_emb_list.append(mean_emb)

        all_esm2_emb = torch.cat(esm2_emb_list, dim=0)  # [N_unique, 2560]
        self.register_buffer("esm2_embeddings", all_esm2_emb.to(self.device))
        self.esm2_id_to_idx = {ensg: i for i, ensg in enumerate(unique_ensg_ids)}

        del esm2_model, esm2_emb_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.print(f"ESM2-3B embeddings: {all_esm2_emb.shape}")

        # ---------------------------------------------------------------- #
        # MLP Architecture                                                   #
        # ---------------------------------------------------------------- #
        hp = self.hparams

        # STRING branch: GNN_DIM (256) → FUSION_DIM (512)
        # Wider projection to match the new 512-dim fusion space
        self.str_proj = nn.Sequential(
            nn.LayerNorm(GNN_DIM),
            nn.Linear(GNN_DIM, FUSION_DIM),  # 256 → 512
        )

        # ESM2-3B branch: ESM2_DIM (2560) → FUSION_DIM (512) via two-layer projection
        # Two-stage compression (2560→1024→512) avoids the 10:1 bottleneck seen in
        # node3-1-1-1-1-2-1 (2560→256 single-layer = same F1 as ESM2-650M, no gain).
        # 5:1 total ratio maintained: 2560/512 = 5.
        self.esm_proj = nn.Sequential(
            nn.LayerNorm(ESM2_DIM),          # LN(2560)
            nn.Linear(ESM2_DIM, 1024),       # 2560 → 1024 (graduated compression)
            nn.GELU(),
            nn.Linear(1024, FUSION_DIM),     # 1024 → 512
        )

        # Gated fusion: (FUSION_DIM, FUSION_DIM) → FUSION_DIM
        # gate_proj: Linear(1024 → 512)
        self.gate_fusion = GatedFusion(in_dim=FUSION_DIM, fusion_dim=FUSION_DIM)

        # Input projection: FUSION_DIM (512) → hidden_dim (384) with trunk regularization
        self.input_proj = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Linear(FUSION_DIM, hp.hidden_dim),  # 512 → 384
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # 3× PreNormResBlock (h=384, inner=768) — proven optimal capacity
        self.blocks = nn.ModuleList([
            PreNormResBlock(hp.hidden_dim, hp.inner_dim, hp.dropout)
            for _ in range(hp.n_blocks)
        ])

        # Output head with head_dropout for dual-modality regularization
        self.output_head = nn.Sequential(
            nn.LayerNorm(hp.hidden_dim),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.hidden_dim, N_GENES * N_CLASSES),
        )

        # Per-gene additive bias: [N_GENES, N_CLASSES] (initialized to zero)
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES, N_CLASSES))

        # Class weights: down=4.77%, neutral=92.82%, up=2.41%
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq) / (1.0 / freq).mean()
        self.register_buffer("class_weights", class_weights)

        # Cast trainable params to float32 for stable optimization
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"node3-1-3-1-1 | ESM2-3B({ESM2_DIM}) + STRING_GNN({GNN_DIM}) → "
            f"GatedFusion({FUSION_DIM}) → {hp.n_blocks}×PreNormResBlock({hp.hidden_dim},{hp.inner_dim}) "
            f"→ HeadDropout({hp.head_dropout}) → Linear→{N_GENES}×{N_CLASSES} + gene_bias | "
            f"Trainable: {trainable:,}/{total:,}"
        )

    # ---------------------------------------------------------------------- #
    # Feature extraction helpers                                              #
    # ---------------------------------------------------------------------- #
    def _get_fused(self, pert_ids: List[str]) -> torch.Tensor:
        """Get gated-fused representation [B, FUSION_DIM] for a batch of ENSG IDs."""
        str_list: List[torch.Tensor] = []
        esm_list: List[torch.Tensor] = []

        for pid in pert_ids:
            # STRING embedding (fallback: zero vector)
            row_str = self.gnn_id_to_idx.get(pid)
            str_list.append(
                self.gnn_embeddings[row_str]
                if row_str is not None
                else torch.zeros(GNN_DIM, device=self.device)
            )
            # ESM2 embedding (fallback: zero vector)
            row_esm = self.esm2_id_to_idx.get(pid)
            esm_list.append(
                self.esm2_embeddings[row_esm]
                if row_esm is not None
                else torch.zeros(ESM2_DIM, device=self.device)
            )

        str_emb = torch.stack(str_list, dim=0).float()   # [B, 256]
        esm_emb = torch.stack(esm_list, dim=0).float()   # [B, 2560]

        h_str = self.str_proj(str_emb)    # [B, 512]
        h_esm = self.esm_proj(esm_emb)    # [B, 512]
        return self.gate_fusion(h_str, h_esm)  # [B, 512]

    def _forward_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        """Forward pass from fused embedding to logits [B, N_CLASSES, N_GENES]."""
        x = self.input_proj(fused)      # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)                # [B, hidden_dim]
        logits = self.output_head(x)    # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)
        # gene_bias: [N_GENES, N_CLASSES].T → [N_CLASSES, N_GENES] → [1, 3, N_GENES]
        return logits + self.gene_bias.T.unsqueeze(0)

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits [B, N_CLASSES, N_GENES]. (Used for val/test)"""
        return self._forward_from_fused(self._get_fused(pert_ids))

    # ---------------------------------------------------------------------- #
    # Manifold Mixup (soft-label KL divergence)                              #
    # ---------------------------------------------------------------------- #
    def _manifold_mixup(
        self, x: torch.Tensor, labels: torch.Tensor
    ):
        """Manifold Mixup in input_proj output space.

        Returns:
            mixed_x: [B, hidden_dim]
            mixed_labels_onehot: [B, N_GENES, N_CLASSES] soft targets
            lam: float
        """
        batch_size = x.size(0)
        lam = float(
            torch.distributions.Beta(
                torch.tensor(self.hparams.mixup_alpha),
                torch.tensor(self.hparams.mixup_alpha),
            ).sample()
        )
        lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5 for clean mixing

        idx = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1.0 - lam) * x[idx]

        labels_onehot = F.one_hot(labels, num_classes=N_CLASSES).float()  # [B, N_GENES, 3]
        mixed_labels_onehot = lam * labels_onehot + (1.0 - lam) * labels_onehot[idx]

        return mixed_x, mixed_labels_onehot, lam

    # ---------------------------------------------------------------------- #
    # Loss                                                                    #
    # ---------------------------------------------------------------------- #
    def _compute_loss_hard(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted cross-entropy with integer labels."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(logits_flat, labels_flat, weight=self.class_weights)

    def _compute_loss_soft(
        self, logits: torch.Tensor, mixed_labels_onehot: torch.Tensor
    ) -> torch.Tensor:
        """Soft-label KL divergence loss for Manifold Mixup.

        logits: [B, N_CLASSES, N_GENES]
        mixed_labels_onehot: [B, N_GENES, N_CLASSES]
        """
        logits_perm = logits.permute(0, 2, 1).float()   # [B, N_GENES, N_CLASSES]
        logits_flat = logits_perm.reshape(-1, N_CLASSES)  # [B*N_GENES, N_CLASSES]
        labels_flat = mixed_labels_onehot.reshape(-1, N_CLASSES)  # [B*N_GENES, N_CLASSES]

        # Weighted soft targets: class-frequency weighting
        weighted_labels = labels_flat * self.class_weights.unsqueeze(0)
        weight_sum = weighted_labels.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weighted_labels = weighted_labels / weight_sum

        log_probs = F.log_softmax(logits_flat, dim=-1)
        return -(weighted_labels * log_probs).sum(dim=-1).mean()

    # ---------------------------------------------------------------------- #
    # Training / Validation / Test steps                                     #
    # ---------------------------------------------------------------------- #
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pert_ids = batch["pert_id"]
        labels = batch["label"]

        fused = self._get_fused(pert_ids)     # [B, FUSION_DIM]
        x = self.input_proj(fused)             # [B, hidden_dim]

        use_mixup = self.training and (torch.rand(1).item() < self.hparams.mixup_prob)
        if use_mixup:
            mixed_x, mixed_labels_oh, _ = self._manifold_mixup(x, labels)
            # Forward through residual blocks and output head (post input_proj)
            for block in self.blocks:
                mixed_x = block(mixed_x)
            logits = self.output_head(mixed_x).view(-1, N_CLASSES, N_GENES)
            logits = logits + self.gene_bias.T.unsqueeze(0)
            loss = self._compute_loss_soft(logits, mixed_labels_oh)
        else:
            for block in self.blocks:
                x = block(x)
            logits = self.output_head(x).view(-1, N_CLASSES, N_GENES)
            logits = logits + self.gene_bias.T.unsqueeze(0)
            loss = self._compute_loss_hard(logits, labels)

        self.log(
            "train/loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        loss = self._compute_loss_hard(logits, batch["label"])
        self.log(
            "val/loss", loss,
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
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

        import torch.distributed as dist

        is_dist = dist.is_available() and dist.is_initialized()

        if is_dist and self.trainer.world_size > 1:
            world_size = dist.get_world_size()
            obj_preds = [None] * world_size
            obj_labels = [None] * world_size
            dist.all_gather_object(obj_preds, preds_local.numpy())
            dist.all_gather_object(obj_labels, labels_local.numpy())
            preds_np = np.concatenate(obj_preds, axis=0)
            labels_np = np.concatenate(obj_labels, axis=0)
        else:
            preds_np = preds_local.numpy()
            labels_np = labels_local.numpy()

        f1 = _compute_per_gene_f1(preds_np, labels_np)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        self._test_preds.append(logits.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist

        preds_local = torch.cat(self._test_preds, dim=0)
        self._test_preds.clear()

        gathered = self.all_gather(preds_local)            # [world_size, N_local, 3, N_GENES]
        all_preds = gathered.view(-1, N_CLASSES, N_GENES)  # [N_total, 3, N_GENES]

        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1
        local_pert_ids = list(self._test_pert_ids)
        local_symbols = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        gathered_pert_ids: List[List[str]] = [local_pert_ids]
        gathered_symbols: List[List[str]] = [local_symbols]
        if world_size > 1 and is_dist:
            obj_pids = [None] * world_size
            obj_syms = [None] * world_size
            dist.all_gather_object(obj_pids, local_pert_ids)
            dist.all_gather_object(obj_syms, local_symbols)
            gathered_pert_ids = obj_pids
            gathered_symbols = obj_syms

        if self.trainer.is_global_zero:
            all_pert_ids = [pid for lst in gathered_pert_ids for pid in lst]
            all_symbols = [sym for lst in gathered_symbols for sym in lst]
            preds_np = all_preds.cpu().numpy()

            # Deduplicate by pert_id (DDP may replicate samples)
            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            for i, pid in enumerate(all_pert_ids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_ids.append(pid)
                    dedup_syms.append(all_symbols[i])
                    dedup_preds.append(preds_np[i])

            # Store for ensemble averaging
            self._final_test_ids = dedup_ids
            self._final_test_syms = dedup_syms
            self._final_test_preds = np.stack(dedup_preds, axis=0)

            _save_test_predictions(
                pert_ids=dedup_ids,
                symbols=dedup_syms,
                preds=np.stack(dedup_preds, axis=0),
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

    # ---------------------------------------------------------------------- #
    # Optimizer: Muon for block weight matrices, AdamW for all else          #
    # ---------------------------------------------------------------------- #
    def configure_optimizers(self):
        hp = self.hparams

        # Muon group: 2D weight matrices in residual block fc1/fc2
        muon_params = [
            p for name, p in self.blocks.named_parameters()
            if p.ndim >= 2 and p.requires_grad and "weight" in name
        ]
        muon_param_ids = {id(p) for p in muon_params}

        # AdamW group: projections, norms, biases, gates, head, gene_bias
        adamw_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in muon_param_ids
        ]

        if MUON_AVAILABLE and muon_params:
            param_groups = [
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=hp.muon_lr,
                    weight_decay=0.0,   # Muon's orthogonalization provides implicit reg
                    momentum=0.95,
                ),
                dict(
                    params=adamw_params,
                    use_muon=False,
                    lr=hp.adamw_lr,
                    betas=(0.9, 0.95),
                    weight_decay=hp.weight_decay,
                ),
            ]
            optimizer = MuonWithAuxAdam(param_groups)
            self.print(
                f"MuonWithAuxAdam: Muon LR={hp.muon_lr} ({len(muon_params)} matrices), "
                f"AdamW LR={hp.adamw_lr} ({len(adamw_params)} params)"
            )
        else:
            self.print("WARNING: Muon unavailable, using pure AdamW fallback")
            optimizer = torch.optim.AdamW(
                [p for p in self.parameters() if p.requires_grad],
                lr=hp.adamw_lr,
                weight_decay=hp.weight_decay,
            )

        # CosineAnnealingWarmRestarts: T_0=80, T_mult=2
        # Cycle lengths: 80, 160, 320 → 3 complete cycles in 560 epochs
        # Avoids RLROP over-triggering that hurt the parent node (6 halvings)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hp.cosine_t0,
            T_mult=hp.cosine_t_mult,
            eta_min=hp.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ---------------------------------------------------------------------- #
    # Checkpoint: save only trainable params + buffers                       #
    # ---------------------------------------------------------------------- #
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {
            prefix + n for n, p in self.named_parameters() if p.requires_grad
        }
        buffer_keys = {prefix + n for n, _ in self.named_buffers()}
        kept = {k: v for k, v in full_sd.items() if k in trainable_keys or k in buffer_keys}

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving checkpoint: {trainable:,}/{total:,} trainable params "
            f"({100 * trainable / total:.2f}%) + {sum(b.numel() for _, b in self.named_buffers()):,} buffer values"
        )
        return kept

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-F1 averaged over all 6640 response genes.

    Exactly matches data/calc_metric.py:
      - argmax(preds, axis=1) → hard predictions
      - Per-gene F1 averaged over present classes only
      - Final = mean over all genes
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [N, N_GENES]
    n_genes = labels.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels[:, g]
        yh = y_hat[:, g]
        per_class_f1 = sk_f1(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        if present.any():
            f1_vals.append(float(per_class_f1[present].mean()))
        else:
            f1_vals.append(0.0)
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    """Save test predictions in TSV format required by calc_metric.py."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "idx": pert_ids[i],
            "input": symbols[i],
            "prediction": json.dumps(preds[i].tolist()),  # [3, 6640] as JSON
        }
        for i in range(len(pert_ids))
    ]
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}", flush=True)


def _ensemble_test_predictions(
    model: PerturbModule,
    datamodule: PerturbDataModule,
    checkpoint_dir: Path,
    output_dir: Path,
    top_k: int = 5,
    threshold: float = 0.003,
) -> None:
    """Load top-K checkpoints and apply threshold-filtered ensemble predictions.

    Selects checkpoints within `threshold` F1 of the best checkpoint, then
    averages logits. This filtered approach outperforms simple top-K averaging
    when checkpoints cluster in quality (proven by tree evidence):
    - node3-1-1-1-1-2-1-1-1 (F1=0.5283): top-3 filtered > top-7 simple average (0.5265)
    - Focuses ensemble on high-quality checkpoints, avoids dilution from lower ones

    Uses rglob("*.ckpt") to correctly find checkpoints stored in nested
    subdirectories (Lightning creates subdirs when filename contains '/').
    Excludes 'last.ckpt' which is not necessarily one of the top-K.
    """
    # Find all checkpoints (excluding last.ckpt)
    all_ckpts = [
        str(p) for p in checkpoint_dir.rglob("*.ckpt")
        if p.name != "last.ckpt"
    ]
    if not all_ckpts:
        print("No checkpoints found for ensemble. Skipping.", flush=True)
        return

    # Sort by val/f1 value embedded in filename if available
    def _parse_f1(path: str) -> float:
        try:
            stem = Path(path).stem
            for part in stem.split("-"):
                if "val_f1" in part or "val/f1" in part:
                    return float(part.split("=")[-1])
            return 0.0
        except Exception:
            return 0.0

    all_ckpts.sort(key=_parse_f1, reverse=True)
    # Start with top-K candidates
    candidate_ckpts = all_ckpts[:top_k]

    # Apply threshold filter: keep checkpoints within `threshold` F1 of the best
    if candidate_ckpts:
        best_f1_val = _parse_f1(candidate_ckpts[0])
        filtered_ckpts = [
            ckpt for ckpt in candidate_ckpts
            if best_f1_val - _parse_f1(ckpt) <= threshold
        ]
        # Ensure at least 1 checkpoint (the best) is always included
        if len(filtered_ckpts) == 0:
            filtered_ckpts = candidate_ckpts[:1]
        selected_ckpts = filtered_ckpts
    else:
        selected_ckpts = candidate_ckpts

    print(
        f"Ensemble: {len(selected_ckpts)} checkpoints (top-{top_k} filtered by ±{threshold} F1):",
        flush=True,
    )
    for p in selected_ckpts:
        print(f"  {Path(p).name} (val/f1={_parse_f1(p):.4f})", flush=True)

    if datamodule.test_ds is None:
        datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    all_ckpt_preds: List[np.ndarray] = []
    ckpt_pert_ids: List[str] = []
    ckpt_symbols: List[str] = []

    for ckpt_path in selected_ckpts:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt)
            model.load_state_dict(sd, strict=False)
            model.eval()

            preds_list: List[torch.Tensor] = []
            ids_list: List[str] = []
            syms_list: List[str] = []

            with torch.no_grad():
                for batch in test_loader:
                    logits = model(batch["pert_id"])
                    preds_list.append(logits.detach().cpu().float())
                    ids_list.extend(batch["pert_id"])
                    syms_list.extend(batch["symbol"])

            ckpt_preds_np = torch.cat(preds_list, dim=0).numpy()
            all_ckpt_preds.append(ckpt_preds_np)
            if not ckpt_pert_ids:
                ckpt_pert_ids = ids_list
                ckpt_symbols = syms_list
            print(f"  Loaded {Path(ckpt_path).name}: {ckpt_preds_np.shape}", flush=True)
        except Exception as e:
            print(f"  Error loading {ckpt_path}: {e}. Skipping.", flush=True)

    if not all_ckpt_preds:
        print("No valid checkpoints loaded. Ensemble aborted.", flush=True)
        return

    avg_preds = np.mean(all_ckpt_preds, axis=0)  # [N_test, 3, N_GENES]

    # Deduplicate by pert_id
    seen: set = set()
    dedup_ids, dedup_syms, dedup_preds = [], [], []
    for i, pid in enumerate(ckpt_pert_ids):
        if pid not in seen:
            seen.add(pid)
            dedup_ids.append(pid)
            dedup_syms.append(ckpt_symbols[i])
            dedup_preds.append(avg_preds[i])

    _save_test_predictions(
        pert_ids=dedup_ids,
        symbols=dedup_syms,
        preds=np.stack(dedup_preds, axis=0),
        out_path=output_dir / "test_predictions.tsv",
    )
    print(
        f"Filtered ensemble predictions saved: {len(dedup_ids)} samples, "
        f"{len(all_ckpt_preds)} checkpoints (threshold={threshold})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="node3-1-3-1-1: ESM2-3B (frozen) + STRING_GNN + FUSION_DIM=512 + "
                    "Gated Fusion + Muon+AdamW + Manifold Mixup (prob=0.75) + CosineWarmRestarts"
    )
    p.add_argument("--micro-batch-size", type=int, default=32,
                   help="Dual-branch: uses more memory, reduce to 16 if OOM")
    p.add_argument("--global-batch-size", type=int, default=256,
                   help="Multiple of micro_batch_size * 8 for DDP compatibility")
    p.add_argument("--max-epochs", type=int, default=500,
                   help="Extended for 3 CosineWarmRestart cycles (80+160+320)")
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--inner-dim", type=int, default=768)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--head-dropout", type=float, default=0.10)
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=8e-4)
    p.add_argument("--cosine-t0", type=int, default=80)
    p.add_argument("--cosine-t-mult", type=int, default=2)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--mixup-prob", type=float, default=0.75,
                   help="Increased from 0.65: tree-best node3-1-1-1-1-2-1-1-1 uses 0.75")
    p.add_argument("--early-stop-patience", type=int, default=25,
                   help="Same as parent: adequate for natural oscillation pattern")
    p.add_argument("--gradient-clip-val", type=float, default=2.0)
    p.add_argument("--save-top-k", type=int, default=5,
                   help="Top-K checkpoints saved; filtered ensemble selects from these")
    p.add_argument("--ensemble-threshold", type=float, default=0.003,
                   help="±F1 threshold for filtered ensemble: keep ckpts within this of best")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--val-check-interval", type=float, default=1.0)
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
    )

    model = PerturbModule(
        hidden_dim=args.hidden_dim,
        inner_dim=args.inner_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay,
        cosine_t0=args.cosine_t0,
        cosine_t_mult=args.cosine_t_mult,
        min_lr=args.min_lr,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
    )

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
        val_check_interval = args.val_check_interval if not fast_dev_run else 1.0
        num_sanity_val_steps = 2

    # IMPORTANT: auto_insert_metric_name=True converts "val/f1" → "val_f1" for the
    # filename, but the metrics dict has "val/f1" (slash). Since the keys don't match,
    # Lightning defaults to 0.0000 for all checkpoints. Fix: use auto_insert_metric_name=False
    # with the actual metric key "val/f1" in the template. Lightning 2.5 handles "/"
    # in filenames as literal characters (no nested dirs with auto_insert_metric_name=False).
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/f1",
        mode="max",
        patience=args.early_stop_patience,
        min_delta=1e-4,  # Require meaningful improvement to reset patience
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(seconds=120),
        ),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        gradient_clip_val=args.gradient_clip_val,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule)

    # ------------------------------------------------------------------ #
    # Test with best checkpoint + Filtered Ensemble                       #
    # ------------------------------------------------------------------ #
    if fast_dev_run or args.debug_max_step is not None:
        trainer.test(model, datamodule=datamodule)
    else:
        # Single-checkpoint test with best checkpoint
        trainer.test(model, datamodule=datamodule, ckpt_path="best")

        # Filtered ensemble: select top-5 checkpoints, filter by ±threshold F1
        # Proven: filtered top-3 (F1=0.5283) > simple top-7 (F1=0.5265) in tree-best node
        if trainer.is_global_zero:
            checkpoint_dir = output_dir / "checkpoints"
            _ensemble_test_predictions(
                model=model,
                datamodule=datamodule,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                top_k=args.save_top_k,
                threshold=args.ensemble_threshold,
            )


if __name__ == "__main__":
    main()
