"""Node 3-3-1-2-1-1-2: Frozen ESM2-650M + STRING_GNN Dual-Branch
              + Gated Fusion + 3-Block Pre-Norm MLP (h=384)
              + Muon Optimizer + Manifold Mixup + Fixed Checkpoint Ensemble
================================================================
Parent  : node3-3-1-2-1-1  (ESM2-150M LoRA, test F1=0.5136)
          - LoRA overfitting on 1,273 samples caused regression from grandparent
          - T_mult=1 compressed cycle (peaked at epoch 87 vs grandparent's 153)
          - 5-checkpoint ensemble only captured declining phase

Grandparent: node3-3-1-2-1 (frozen ESM2-150M, test F1=0.5170)
          - Second-best in entire tree (only 0.0008 behind tree-best 0.5178)
          - Best checkpoint at epoch 153 (val F1=0.5163)

Key changes from parent node3-3-1-2-1-1
--------------------------------------
1. FROZEN ESM2-650M (replacing LoRA ESM2-150M):
   The parent's LoRA fine-tuning on ESM2-150M caused severe overfitting on the
   1,273-sample training set: train loss dropped from 0.423 → 0.336 while val F1
   fell from 0.5141 → 0.5080 — classic overfitting. The feedback explicitly
   recommends reverting to frozen pre-trained representations for better
   generalization.

   We scale from ESM2-150M (640-dim) to ESM2-650M (1280-dim) to exploit richer
   protein representations without fine-tuning risk. The tree-best node
   (node4-1-1-1-1-1-1, F1=0.5178) uses ESM2-650M LoRA, confirming 650M > 150M.
   Frozen ESM2-650M precomputed at setup() gives efficient inference with no
   overfitting.

2. RESTORED T_MULT=2 (from parent's T_mult=1):
   Parent's T_mult=1 created 80-epoch uniform cycles; model peaked at epoch 87
   (end of cycle 1) and never recovered. The grandparent used T_mult=2 and
   peaked at epoch 153 (end of cycle 1 = 80 epochs, well into cycle 2 = 160).
   T_mult=2 creates longer cycles that match the natural convergence of this
   architecture: 80 → 160 epochs, giving the model sufficient time to find
   sharp minima before restarting.

3. RESTORED MIXUP PROB=0.65 AND HEAD_DROPOUT=0.10:
   Grandparent's successful configuration with frozen ESM2-150M. The parent
   over-regularized (0.65 → 0.50 mixup, 0.10 → 0.05 head_dropout) to compensate
   for LoRA overfitting, but with frozen ESM2-650M these reductions are no longer
   needed. Grandparent's Mixup prob=0.65 + head_dropout=0.10 achieved F1=0.5170.

4. EXTENDED TRAINING: max_epochs=600, patience=80
   600 epochs / T_mult=2 schedule: cycle 1 = 80, cycle 2 = 160, cycle 3 = 320.
   patience=80 = 1 full first cycle (enough time to recover after each restart).

Architecture:
--------------------------------------------
Perturbed Gene (ENSG ID)
     │
     ├─── STRING_GNN frozen [18,870 × 256] ──→ LN → Linear(256→256) → h_str
     │
     └─── ESM2-650M frozen (precomputed at setup)
              → mean_pool → [B, 1280] → LN → Linear(1280→256) → h_esm

     gate = sigmoid(Linear(512 → 256)([h_str, h_esm]))
     fused = gate * h_str + (1 - gate) * h_esm    → [B, 256]

     LN(256) → Linear(256→384) → GELU → Dropout(0.30)   [input_proj]
          ↓ [Manifold Mixup applied here during training, prob=0.65]
     3×PreNormResBlock(h=384, inner=768, dropout=0.30)
     LN(384) → Dropout(0.10) → Linear(384 → 19,920) + gene_bias

Training: Muon (LR=0.01) for MLP block 2D weight matrices +
          AdamW (LR=3e-4) for projections + fusion + head + gene_bias;
          CosineAnnealingWarmRestarts(T_0=80, T_mult=2);
          Weighted CE; Manifold Mixup (alpha=0.2, prob=0.65).

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRING_GNN_DIR = "/home/Models/STRING_GNN"
PROTEIN_FASTA = "/home/data/genome/hg38_gencode_protein.fa"
ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ESM2_DIM = 1280      # ESM2-650M hidden dimension (vs 640 for ESM2-150M)
GNN_DIM = 256        # STRING_GNN output embedding dimension
FUSION_DIM = 256     # Gated fusion output dimension
HIDDEN_DIM = 384     # MLP hidden dimension — proven optimal
INNER_DIM = 768      # MLP inner (expansion) dimension (2x hidden per PreLN block)
N_GENES = 6640       # number of response genes per perturbation
N_CLASSES = 3        # down (-1→0), neutral (0→1), up (1→2)

# Fallback protein sequence for genes not found in FASTA
FALLBACK_SEQ = "MAAAAA"


# ---------------------------------------------------------------------------
# Protein sequence loading from FASTA
# ---------------------------------------------------------------------------
def _build_ensg_to_seq(fasta_path: str) -> Dict[str, str]:
    """Build ENSG→longest protein sequence map from GENCODE protein FASTA."""
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
                # Parse ENSG ID from header like:
                # >ENSP00000... gene:ENSG00000... transcript:...
                header = line[1:]
                current_ensg = None
                for part in header.split():
                    if part.startswith("gene:ENSG"):
                        current_ensg = part.split(":")[1]
                        break
            else:
                current_seq_parts.append(line)
    flush()

    # Keep the longest isoform per gene
    return {
        ensg: max(seqs, key=len)
        for ensg, seqs in ensg2seqs.items()
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Each sample is one gene perturbation experiment in HepG2 cells."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        if "label" in df.columns:
            # Labels in {-1,0,1} → shift to {0,1,2}
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
    """Pre-LayerNorm residual block.

    Architecture:
        output = x + LN(x) → Linear(dim→inner) → GELU → Dropout
                               → Linear(inner→dim) → Dropout
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


class GatedFusion(nn.Module):
    """Learnable gated fusion of two embedding branches.

    Given STRING branch h_str (FUSION_DIM) and ESM2 branch h_esm (FUSION_DIM):
        gate = sigmoid(Linear(concat([h_str, h_esm]) → fusion_dim))
        fused = gate * h_str + (1 - gate) * h_esm
    """

    def __init__(self, in_dim: int = 256, fusion_dim: int = 256) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(in_dim * 2, fusion_dim)
        self._in_dim = in_dim

    def forward(self, h_str: torch.Tensor, h_esm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_str: [B, in_dim] — STRING branch projected features
            h_esm: [B, in_dim] — ESM2 branch projected features
        Returns:
            fused: [B, fusion_dim]
        """
        gate = torch.sigmoid(self.gate_proj(torch.cat([h_str, h_esm], dim=-1)))
        return gate * h_str + (1.0 - gate) * h_esm


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        inner_dim: int = INNER_DIM,
        n_blocks: int = 3,              # Proven 3-block architecture
        dropout: float = 0.30,          # Trunk dropout
        head_dropout: float = 0.10,     # RESTORED from grandparent: dual-modality head dropout
        muon_lr: float = 0.01,          # Proven Muon LR
        adamw_lr: float = 3e-4,         # AdamW LR for non-block params
        weight_decay: float = 0.01,     # Proven weight decay
        label_smoothing: float = 0.0,   # No smoothing (proven best)
        cosine_t0: int = 80,            # CosineWarmRestarts first cycle length
        cosine_t_mult: int = 2,         # RESTORED to 2: longer cycles for proper convergence
        min_lr: float = 1e-7,
        grad_clip_norm: float = 2.0,    # Proven clip for Muon
        mixup_alpha: float = 0.2,       # Manifold Mixup interpolation coefficient
        mixup_prob: float = 0.65,       # RESTORED: grandparent's successful Mixup prob
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Populated in setup()
        self.str_proj: Optional[nn.Sequential] = None   # STRING branch projection
        self.esm_proj: Optional[nn.Sequential] = None   # ESM2 branch projection
        self.gate_fusion: Optional[GatedFusion] = None  # Gated fusion
        self.input_proj: Optional[nn.Sequential] = None
        self.blocks: Optional[nn.ModuleList] = None
        self.output_head: Optional[nn.Sequential] = None
        self.gene_bias: Optional[nn.Parameter] = None

        # STRING_GNN gene-ID → embedding-row index
        self.gnn_id_to_idx: Dict[str, int] = {}
        # ESM2-650M: ENSG-ID → precomputed embedding index
        self.esm2_id_to_idx: Dict[str, int] = {}

        # Metric accumulators
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        """Build model: STRING_GNN frozen buffer + ESM2-650M precomputed + MLP architecture."""
        import torch.distributed as dist
        from transformers import AutoTokenizer, EsmForMaskedLM

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # ---- STRING_GNN frozen embeddings ----
        self.print("Loading STRING_GNN and computing frozen node embeddings ...")
        gnn_model = __import__("transformers").AutoModel.from_pretrained(
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

        # Register as a non-trainable float32 buffer [18870, 256]
        all_emb = gnn_out.last_hidden_state.detach().float()
        self.register_buffer("gnn_embeddings", all_emb)

        # Free GNN model memory
        del gnn_model, gnn_out, graph, edge_index, edge_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.print(f"STRING_GNN embeddings shape: {all_emb.shape}")

        # Build ENSG-ID → row-index mapping for STRING_GNN
        node_names: List[str] = json.loads(
            (Path(STRING_GNN_DIR) / "node_names.json").read_text()
        )
        self.gnn_id_to_idx = {name: i for i, name in enumerate(node_names)}
        self.print(f"STRING_GNN covers {len(self.gnn_id_to_idx)} Ensembl gene IDs")

        # ---- Load protein FASTA for ESM2-650M inference ----
        self.print(f"Building ENSG → protein sequence map from {PROTEIN_FASTA} ...")
        ensg2seq = _build_ensg_to_seq(PROTEIN_FASTA)
        self.print(f"FASTA contains {len(ensg2seq)} ENSG → protein sequence entries")

        # ---- Precompute frozen ESM2-650M embeddings ----
        # Unlike parent node (LoRA fine-tuning), ESM2-650M is used frozen.
        # Precomputing at setup() is more memory-efficient than on-the-fly inference.
        self.print(f"Loading ESM2-650M tokenizer (rank 0 downloads first) ...")

        if local_rank == 0:
            AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)

        self.print(f"Loading ESM2-650M base model for frozen embeddings ...")
        if local_rank == 0:
            EsmForMaskedLM.from_pretrained(ESM2_MODEL_NAME)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Load ESM2-650M in float32 for precomputation
        esm2_model = EsmForMaskedLM.from_pretrained(ESM2_MODEL_NAME, dtype=torch.float32)
        esm2_model.eval()
        esm2_model = esm2_model.to(self.device)

        # Collect all unique ENSG IDs needed (from all splits accessible at this point)
        # We collect from both GNN node names and FASTA to be comprehensive.
        # The actual unique IDs will be determined at inference time.
        # Here we precompute for all IDs present in the FASTA file.
        all_ensg_ids = sorted(ensg2seq.keys())
        # Add any STRING GNN IDs not in FASTA (will use fallback sequence)
        all_ensg_ids_set = set(all_ensg_ids)
        for gnn_id in node_names:
            if gnn_id not in all_ensg_ids_set:
                all_ensg_ids.append(gnn_id)
                all_ensg_ids_set.add(gnn_id)

        self.print(f"Precomputing ESM2-650M embeddings for {len(all_ensg_ids)} ENSG IDs ...")

        esm2_embeddings_list = []
        batch_size_esm = 8  # Small batch for ESM2-650M to avoid OOM
        with torch.no_grad():
            for i in range(0, len(all_ensg_ids), batch_size_esm):
                batch_ids = all_ensg_ids[i:i + batch_size_esm]
                seqs = [ensg2seq.get(pid, FALLBACK_SEQ)[:1022] for pid in batch_ids]

                tokenized = tokenizer(
                    seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                )
                tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

                outputs = esm2_model(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    output_hidden_states=True,
                )

                # Mean-pool last hidden state, excluding special tokens (cls, eos, pad)
                hidden = outputs.hidden_states[-1]  # [B, seq_len, 1280]
                special_ids_tensor = torch.tensor(
                    [
                        tokenizer.pad_token_id,
                        tokenizer.cls_token_id,
                        tokenizer.eos_token_id,
                    ],
                    device=self.device,
                )
                special_mask = torch.isin(tokenized["input_ids"], special_ids_tensor)
                masked_states = hidden.masked_fill(special_mask.unsqueeze(-1), 0.0)
                token_counts = (~special_mask).float().sum(dim=1, keepdim=True).clamp(min=1e-9)
                mean_embeddings = masked_states.sum(dim=1) / token_counts  # [B, 1280]

                esm2_embeddings_list.append(mean_embeddings.detach().cpu().float())

                if (i // batch_size_esm) % 50 == 0:
                    self.print(f"  ESM2-650M: processed {i + len(batch_ids)}/{len(all_ensg_ids)} proteins")

        # Register as non-trainable buffer [N_ensg, 1280]
        esm2_all_emb = torch.cat(esm2_embeddings_list, dim=0)  # [N_ensg, 1280]
        self.register_buffer("esm2_embeddings", esm2_all_emb)

        # Build ENSG-ID → row-index mapping for ESM2 embeddings
        self.esm2_id_to_idx = {ensg: i for i, ensg in enumerate(all_ensg_ids)}
        self.print(f"ESM2-650M frozen embeddings: {esm2_all_emb.shape}")

        # Free ESM2 model memory after precomputation
        del esm2_model, esm2_embeddings_list, ensg2seq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- MLP architecture ----
        hp = self.hparams

        # STRING branch: project GNN_DIM → FUSION_DIM
        self.str_proj = nn.Sequential(
            nn.LayerNorm(GNN_DIM),
            nn.Linear(GNN_DIM, FUSION_DIM),
        )

        # ESM2-650M branch: project ESM2_DIM (1280) → FUSION_DIM (256)
        self.esm_proj = nn.Sequential(
            nn.LayerNorm(ESM2_DIM),
            nn.Linear(ESM2_DIM, FUSION_DIM),
        )

        # Gated fusion: (FUSION_DIM, FUSION_DIM) → FUSION_DIM
        self.gate_fusion = GatedFusion(in_dim=FUSION_DIM, fusion_dim=FUSION_DIM)

        # Input projection: FUSION_DIM → hidden_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Linear(FUSION_DIM, hp.hidden_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # 3× PreNormResBlock (h=384, inner=768)
        self.blocks = nn.ModuleList(
            [
                PreNormResBlock(hp.hidden_dim, hp.inner_dim, hp.dropout)
                for _ in range(hp.n_blocks)
            ]
        )

        # Output head — RESTORED head_dropout=0.10 (grandparent's successful setting)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hp.hidden_dim),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.hidden_dim, N_GENES * N_CLASSES),
        )

        # Per-gene additive bias: one offset per (gene × class) pair
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES, N_CLASSES))

        # ---- Class weights (CORRECT ordering after +1 label shift) ----
        # class 0 = down-regulated  (4.77%)  → high weight
        # class 1 = neutral         (92.82%) → low weight
        # class 2 = up-regulated    (2.41%)  → highest weight
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
            f"Architecture: frozen ESM2-650M({ESM2_DIM}) + STRING_GNN({GNN_DIM}) → "
            f"GatedFusion({FUSION_DIM}) → ManifoldMixup → "
            f"{hp.n_blocks}×PreNormResBlock({hp.hidden_dim},{hp.inner_dim}) "
            f"→ HeadDropout({hp.head_dropout}) → Linear({hp.hidden_dim},{N_GENES}×{N_CLASSES}) + gene_bias"
        )
        self.print(f"Trainable params: {trainable:,} / {total:,}")

    # ------------------------------------------------------------------
    def _get_esm2_embedding(self, pert_ids: List[str]) -> torch.Tensor:
        """Get frozen ESM2-650M embeddings from precomputed buffer.

        Unlike parent node (LoRA on-the-fly), this uses frozen precomputed
        embeddings from the buffer. No gradient flows through ESM2 weights.

        Args:
            pert_ids: List of ENSG IDs for the batch

        Returns:
            embeddings: [B, ESM2_DIM=1280] float32 tensor
        """
        emb_list: List[torch.Tensor] = []
        for pid in pert_ids:
            row = self.esm2_id_to_idx.get(pid)
            if row is not None:
                emb_list.append(self.esm2_embeddings[row])
            else:
                # Fallback: zero embedding for unknown genes
                emb_list.append(torch.zeros(ESM2_DIM, device=self.device, dtype=torch.float32))
        return torch.stack(emb_list, dim=0)  # [B, 1280]

    def _get_str_embedding(self, pert_ids: List[str]) -> torch.Tensor:
        """Get frozen STRING_GNN embeddings for a batch of ENSG IDs."""
        str_list: List[torch.Tensor] = []
        for pid in pert_ids:
            row = self.gnn_id_to_idx.get(pid)
            if row is not None:
                str_list.append(self.gnn_embeddings[row])
            else:
                str_list.append(torch.zeros(GNN_DIM, device=self.device, dtype=torch.float32))
        return torch.stack(str_list, dim=0)  # [B, 256]

    def _get_fused(self, pert_ids: List[str]) -> torch.Tensor:
        """Get gated fused representation for a batch of ENSG IDs.

        Combines frozen STRING_GNN topology embeddings (256-dim) with
        frozen ESM2-650M protein sequence embeddings (1280-dim) via a
        learnable sigmoidal gate.

        Returns:
            fused: [B, FUSION_DIM=256]
        """
        str_emb = self._get_str_embedding(pert_ids)   # [B, 256]
        esm_emb = self._get_esm2_embedding(pert_ids)  # [B, 1280]

        h_str = self.str_proj(str_emb)   # [B, 256]
        h_esm = self.esm_proj(esm_emb)   # [B, 256]
        fused = self.gate_fusion(h_str, h_esm)  # [B, 256]
        return fused

    def _manifold_mixup(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Apply Manifold Mixup: interpolate hidden representations and labels.

        Args:
            x: [B, hidden_dim] - features after input_proj
            labels: [B, N_GENES] - integer class labels in {0, 1, 2}

        Returns:
            mixed_x: [B, hidden_dim] - mixed features
            mixed_labels_onehot: [B, N_GENES, N_CLASSES] - soft mixed one-hot labels
            lam: float - mixing coefficient
        """
        hp = self.hparams
        batch_size = x.size(0)

        # Sample mixing coefficient from Beta distribution
        lam = float(torch.distributions.Beta(
            torch.tensor(hp.mixup_alpha),
            torch.tensor(hp.mixup_alpha)
        ).sample())
        lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5 for clean mixing

        # Random permutation within batch
        idx = torch.randperm(batch_size, device=x.device)

        # Mix features
        mixed_x = lam * x + (1.0 - lam) * x[idx]

        # Convert integer labels to one-hot float
        labels_onehot = F.one_hot(labels, num_classes=N_CLASSES).float()  # [B, N_GENES, 3]
        mixed_labels_onehot = lam * labels_onehot + (1.0 - lam) * labels_onehot[idx]

        return mixed_x, mixed_labels_onehot, lam

    def _forward_from_proj(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through blocks and output head (after input_proj)."""
        for block in self.blocks:
            x = block(x)                               # [B, hidden_dim]
        logits = self.output_head(x)                   # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)   # [B, 3, 6640]
        # gene_bias: [N_GENES, N_CLASSES].T → [N_CLASSES, N_GENES] → [1, 3, N_GENES]
        logits = logits + self.gene_bias.T.unsqueeze(0)
        return logits

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits of shape [B, N_CLASSES, N_GENES]. (Used for val/test)"""
        fused = self._get_fused(pert_ids)             # [B, 256]
        x = self.input_proj(fused)                    # [B, hidden_dim]
        return self._forward_from_proj(x)             # [B, 3, 6640]

    def _compute_loss_hard(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted CE on [B, N_CLASSES, N_GENES] logits with integer labels."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    def _compute_loss_soft(
        self, logits: torch.Tensor, mixed_labels_onehot: torch.Tensor
    ) -> torch.Tensor:
        """Soft-label cross-entropy for Manifold Mixup.

        Args:
            logits: [B, N_CLASSES, N_GENES]
            mixed_labels_onehot: [B, N_GENES, N_CLASSES] soft one-hot labels

        Returns:
            scalar loss
        """
        logits_perm = logits.permute(0, 2, 1).float()  # [B, N_GENES, 3]

        # Flatten both to [B*N_GENES, N_CLASSES]
        logits_flat = logits_perm.reshape(-1, N_CLASSES)
        labels_flat = mixed_labels_onehot.reshape(-1, N_CLASSES)  # [B*N_GENES, 3]

        # Apply class weights (broadcast over batch×gene dimension)
        weighted_labels = labels_flat * self.class_weights.unsqueeze(0)  # [B*N_GENES, 3]
        weight_sum = weighted_labels.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weighted_labels = weighted_labels / weight_sum

        # KL divergence as soft cross-entropy: -sum(soft_target * log_softmax(logit))
        log_probs = F.log_softmax(logits_flat, dim=-1)
        loss = -(weighted_labels * log_probs).sum(dim=-1).mean()

        return loss

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pert_ids = batch["pert_id"]
        labels = batch["label"]

        # Get gated fused embedding (frozen ESM2-650M + frozen STRING_GNN)
        fused = self._get_fused(pert_ids)  # [B, 256]
        x = self.input_proj(fused)         # [B, hidden_dim]

        hp = self.hparams
        # Apply Manifold Mixup with probability mixup_prob (0.65 — restored from grandparent)
        use_mixup = self.training and (torch.rand(1).item() < hp.mixup_prob)

        if use_mixup:
            mixed_x, mixed_labels_onehot, lam = self._manifold_mixup(x, labels)
            logits = self._forward_from_proj(mixed_x)
            loss = self._compute_loss_soft(logits, mixed_labels_onehot)
        else:
            logits = self._forward_from_proj(x)
            loss = self._compute_loss_hard(logits, labels)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        loss = self._compute_loss_hard(logits, batch["label"])
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        preds_local = torch.cat(self._val_preds, dim=0)    # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0)  # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        import torch.distributed as dist

        is_dist = dist.is_available() and dist.is_initialized()

        if is_dist and self.trainer.world_size > 1:
            world_size = dist.get_world_size()
            preds_np_local = preds_local.numpy()
            labels_np_local = labels_local.numpy()

            obj_preds = [None] * world_size
            obj_labels = [None] * world_size
            dist.all_gather_object(obj_preds, preds_np_local)
            dist.all_gather_object(obj_labels, labels_np_local)

            preds_np = np.concatenate(obj_preds, axis=0)
            labels_np = np.concatenate(obj_labels, axis=0)

            f1 = _compute_per_gene_f1(preds_np, labels_np)
            self.log("val/f1", f1, prog_bar=True, sync_dist=True)
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

        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        self._test_preds.clear()

        # self.all_gather always prepends world_size dim
        gathered = self.all_gather(preds_local)  # [world_size, N_local, 3, 6640]
        all_preds = gathered.view(-1, N_CLASSES, N_GENES)  # [N_total, 3, 6640]

        # Gather string metadata
        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1
        local_pert_ids = list(self._test_pert_ids)
        local_symbols = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        gathered_pert_ids_flat: List[List[str]] = [local_pert_ids]
        gathered_symbols_flat: List[List[str]] = [local_symbols]
        if world_size > 1 and is_dist:
            obj_pids = [None] * world_size
            obj_syms = [None] * world_size
            dist.all_gather_object(obj_pids, local_pert_ids)
            dist.all_gather_object(obj_syms, local_symbols)
            gathered_pert_ids_flat = obj_pids
            gathered_symbols_flat = obj_syms

        # Only global rank 0 saves test predictions to avoid duplicate writes
        global_rank = dist.get_rank() if (is_dist and dist.is_initialized()) else 0
        if global_rank == 0:
            all_pert_ids = [pid for lst in gathered_pert_ids_flat for pid in lst]
            all_symbols = [sym for lst in gathered_symbols_flat for sym in lst]

            # De-duplicate (DDP may replicate samples across ranks)
            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            preds_np = all_preds.cpu().numpy()  # [N_total, 3, 6640]
            for i, pid in enumerate(all_pert_ids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_ids.append(pid)
                    dedup_syms.append(all_symbols[i])
                    dedup_preds.append(preds_np[i])

            # Store for potential ensemble averaging later
            self._final_test_ids = dedup_ids
            self._final_test_syms = dedup_syms
            self._final_test_preds = np.stack(dedup_preds, axis=0)

            _save_test_predictions(
                pert_ids=dedup_ids,
                symbols=dedup_syms,
                preds=np.stack(dedup_preds, axis=0),
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        from muon import MuonWithAuxAdam

        hp = self.hparams

        # Separate parameters for Muon vs AdamW:
        # Muon: 2D weight matrices in the hidden residual MLP blocks
        # AdamW: ALL other trainable params including:
        #   - STRING branch projection (str_proj)
        #   - ESM2 branch projection (esm_proj)
        #   - Gated fusion (gate_fusion)
        #   - Input projection (input_proj)
        #   - Output head (output_head)
        #   - Per-gene bias (gene_bias)
        muon_params = [
            p for name, p in self.blocks.named_parameters()
            if p.ndim >= 2 and p.requires_grad
        ]

        # All other trainable params go to AdamW
        muon_param_ids = set(id(p) for p in muon_params)
        adamw_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in muon_param_ids
        ]

        n_muon = sum(p.numel() for p in muon_params)
        n_adamw = sum(p.numel() for p in adamw_params)
        self.print(
            f"Optimizer split: Muon={n_muon:,} params (MLP 2D weights), "
            f"AdamW={n_adamw:,} params (projections + fusion + head + bias)"
        )

        param_groups = [
            # Muon group for hidden block weight matrices
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.weight_decay,
                momentum=0.95,
            ),
            # AdamW group for projections + norms + biases + gates + head
            dict(
                params=adamw_params,
                use_muon=False,
                lr=hp.adamw_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.weight_decay,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # CosineAnnealingWarmRestarts: T_0=80, T_mult=2 (RESTORED from grandparent)
        # Grandparent's T_mult=2 created cycles: 80 → 160 → 320 epochs.
        # Grandparent peaked at epoch 153 (inside cycle 2 = 160 epochs).
        # T_mult=1's uniform 80-epoch cycles were too short (parent peaked at epoch 87).
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

    # ------------------------------------------------------------------
    # Checkpoint helpers (save only trainable params + small buffers)
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and essential buffers.

        With frozen ESM2-650M + frozen STRING_GNN, only the MLP params
        are trainable (~9.9M). The large buffers (gnn_embeddings ~18870×256,
        esm2_embeddings ~N_proteins×1280) are reconstructed at setup().
        """
        full_state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        trainable_state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_state_dict:
                    trainable_state_dict[key] = full_state_dict[key]

        # Also save essential small buffers (class_weights)
        # Exclude large frozen buffers (gnn_embeddings, esm2_embeddings) — rebuilt at setup()
        small_buffers = {"class_weights"}
        for name, buffer in self.named_buffers():
            if name in small_buffers:
                key = prefix + name
                if key in full_state_dict:
                    trainable_state_dict[key] = full_state_dict[key]

        self.print(
            f"Saving checkpoint: {len(trainable_state_dict)} tensors, "
            f"~{sum(v.numel() for v in trainable_state_dict.values()):,} elements"
        )

        return trainable_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """Load trainable params from checkpoint (strict=False handles frozen buffers)."""
        full_state_keys = set(super().state_dict().keys())
        trainable_keys = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        small_buffer_keys = {"class_weights"}
        expected_keys = trainable_keys | small_buffer_keys

        missing_keys = [k for k in expected_keys if k not in state_dict]
        unexpected_keys = [k for k in state_dict if k not in expected_keys]

        if missing_keys:
            self.print(f"Missing checkpoint keys: {len(missing_keys)} (expected for frozen buffers)")
        if unexpected_keys:
            self.print(f"Unexpected checkpoint keys: {unexpected_keys[:5]}...")

        loaded_trainable = len([k for k in state_dict if k in trainable_keys])
        loaded_buffers = len([k for k in state_dict if k in small_buffer_keys])
        self.print(
            f"Loading checkpoint: {loaded_trainable} trainable params + {loaded_buffers} buffers"
        )

        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-averaged F1 exactly matching calc_metric.py.

    preds  : [N_samples, 3, N_genes]  — logits / class scores
    labels : [N_samples, N_genes]     — integer class labels in {0, 1, 2}
    """
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
    """Save test predictions in the TSV format required by calc_metric.py."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert len(pert_ids) == len(preds), (
        f"Length mismatch: {len(pert_ids)} pert_ids vs {len(preds)} pred rows"
    )
    rows = [
        {
            "idx": pert_ids[i],
            "input": symbols[i],
            "prediction": json.dumps(preds[i].tolist()),  # [3, 6640] as JSON
        }
        for i in range(len(pert_ids))
    ]
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}")


def _ensemble_test_predictions(
    model: PerturbModule,
    datamodule: PerturbDataModule,
    checkpoint_paths: List[str],
    output_dir: Path,
) -> None:
    """Load top-K checkpoints and average logits for final test predictions.

    With frozen ESM2-650M, all checkpoints share the same precomputed
    ESM2 embeddings (stored in buffer). Loading each checkpoint only
    updates the MLP params, ensuring consistent embedding usage.
    """
    if not checkpoint_paths:
        print("No checkpoints found for ensemble. Skipping ensemble.")
        return

    print(f"Running checkpoint ensemble with {len(checkpoint_paths)} checkpoints...")
    print("Checkpoints:", [Path(p).name for p in checkpoint_paths])

    # Ensure the test dataloader is set up
    if datamodule.test_ds is None:
        datamodule.setup("test")

    test_loader = datamodule.test_dataloader()
    all_checkpoint_preds = []  # List of [N_test, 3, N_genes] arrays

    for ckpt_path in checkpoint_paths:
        try:
            # Load checkpoint state dict
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt

            # Load into model (strict=False handles frozen buffer keys not in ckpt)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            target_device = next(
                (p.device for p in model.parameters() if p.requires_grad),
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

            ckpt_preds = []
            ckpt_pert_ids = []
            ckpt_symbols = []

            with torch.no_grad():
                for batch in test_loader:
                    logits = model(batch["pert_id"])
                    ckpt_preds.append(logits.detach().cpu().float())
                    ckpt_pert_ids.extend(batch["pert_id"])
                    ckpt_symbols.extend(batch["symbol"])

            ckpt_preds_np = torch.cat(ckpt_preds, dim=0).numpy()  # [N_test, 3, N_genes]
            all_checkpoint_preds.append(ckpt_preds_np)
            print(f"  Loaded: {Path(ckpt_path).name}, shape: {ckpt_preds_np.shape}")

        except Exception as e:
            print(f"  Error loading checkpoint {ckpt_path}: {e}. Skipping.")

    if not all_checkpoint_preds:
        print("No valid checkpoints loaded. Ensemble aborted.")
        return

    # Average logits across all checkpoints
    avg_preds = np.mean(all_checkpoint_preds, axis=0)  # [N_test, 3, N_genes]

    # De-duplicate based on pert_id
    seen: set = set()
    dedup_ids, dedup_syms, dedup_preds = [], [], []
    for i, pid in enumerate(ckpt_pert_ids):
        if pid not in seen:
            seen.add(pid)
            dedup_ids.append(pid)
            dedup_syms.append(ckpt_symbols[i])
            dedup_preds.append(avg_preds[i])

    ensemble_preds = np.stack(dedup_preds, axis=0)  # [N_test_unique, 3, N_genes]

    ensemble_path = output_dir / "test_predictions.tsv"
    _save_test_predictions(
        pert_ids=dedup_ids,
        symbols=dedup_syms,
        preds=ensemble_preds,
        out_path=ensemble_path,
    )
    print(
        f"Ensemble predictions saved: {ensemble_path} "
        f"({len(dedup_ids)} samples, {len(all_checkpoint_preds)} checkpoints)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Node3-3-1-2-1-1-2: Frozen ESM2-650M + STRING_GNN Dual-Branch + "
            "3-Block MLP (h=384) + Muon + Mixup(prob=0.65) + Ensemble"
        )
    )
    p.add_argument("--micro-batch-size", type=int, default=32,
                   help="Micro-batch size per GPU")
    p.add_argument("--global-batch-size", type=int, default=256,
                   help="Global batch size (multiple of micro_batch_size * 8)")
    p.add_argument("--max-epochs", type=int, default=600,
                   help="Extended to 600: T_mult=2 gives 80→160→320 cycles")
    p.add_argument("--muon-lr", type=float, default=0.01,
                   help="Muon LR for MLP block 2D weight matrices")
    p.add_argument("--adamw-lr", type=float, default=3e-4,
                   help="AdamW LR for all non-Muon params")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.30, help="Trunk dropout")
    p.add_argument("--head-dropout", type=float, default=0.10,
                   help="RESTORED from grandparent: output head dropout")
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--inner-dim", type=int, default=768)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--cosine-t0", type=int, default=80,
                   help="CosineWarmRestarts first cycle length")
    p.add_argument("--cosine-t-mult", type=int, default=2,
                   help="RESTORED from grandparent: T_mult=2 for 80→160→320 cycles")
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--grad-clip-norm", type=float, default=2.0)
    p.add_argument("--mixup-alpha", type=float, default=0.2,
                   help="Beta distribution alpha for Manifold Mixup")
    p.add_argument("--mixup-prob", type=float, default=0.65,
                   help="RESTORED from grandparent: probability of applying Mixup per batch")
    p.add_argument("--save-top-k", type=int, default=5,
                   help="Number of top checkpoints to save for ensemble")
    p.add_argument("--early-stop-patience", type=int, default=80,
                   help="80 epochs = 1 full first cycle: allows recovery after warm restart")
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
        label_smoothing=args.label_smoothing,
        cosine_t0=args.cosine_t0,
        cosine_t_mult=args.cosine_t_mult,
        min_lr=args.min_lr,
        grad_clip_norm=args.grad_clip_norm,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
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

    # Use slash-free filename to avoid Lightning creating subdirectories.
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-val_f1={val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,  # Prevents Lightning from adding "/" to filename
    )
    early_stop_cb = EarlyStopping(
        monitor="val/f1",
        mode="max",
        patience=args.early_stop_patience,  # 80 epochs = 1 full first cycle
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
        strategy=DDPStrategy(
            find_unused_parameters=False,  # All parameters are used in forward pass
            timeout=timedelta(seconds=120),
        ),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval if (
            args.debug_max_step is None and not fast_dev_run
        ) else 1.0,
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
    # Test: Run best checkpoint, then ensemble if multiple checkpoints exist.
    # ------------------------------------------------------------------
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        # Single best checkpoint test first (standard Lightning approach)
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

        # Checkpoint ensemble: collect all saved top-k checkpoints and average logits.
        # Use rglob to find all .ckpt files (handles any nested directory structure).
        if trainer.is_global_zero:
            ckpt_dir = output_dir / "checkpoints"

            # rglob finds ALL .ckpt files recursively; filter out "last.ckpt"
            ckpt_paths_raw = [
                p for p in ckpt_dir.rglob("*.ckpt")
                if "last" not in p.name
            ]

            # Sort by F1 value in filename (higher is better)
            # Filenames like: best-001-val_f1=0.5163.ckpt
            def _extract_f1(path_obj: Path) -> float:
                try:
                    parts = path_obj.stem.split("=")
                    return float(parts[-1])
                except (ValueError, IndexError):
                    return 0.0

            ckpt_paths_sorted = sorted(ckpt_paths_raw, key=_extract_f1, reverse=True)
            ckpt_paths = [str(p) for p in ckpt_paths_sorted[:args.save_top_k]]

            if len(ckpt_paths) > 1:
                print(
                    f"Found {len(ckpt_paths)} checkpoints for ensemble: "
                    f"{[Path(p).name for p in ckpt_paths]}"
                )

                # Disable dropout for ensemble inference
                model.eval()
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = 0.0

                # Move model to single GPU for ensemble
                model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = model.to(model_device)

                # Re-setup data for single-device inference
                datamodule.setup("test")
                _ensemble_test_predictions(
                    model=model,
                    datamodule=datamodule,
                    checkpoint_paths=ckpt_paths,
                    output_dir=output_dir,
                )
            else:
                print(
                    f"Only {len(ckpt_paths)} checkpoint(s) found — skipping ensemble, "
                    "single-checkpoint predictions already saved."
                )

    if trainer.is_global_zero:
        print(f"\nTraining complete. Test results: {test_results}")
        print(f"Predictions saved to: {output_dir / 'test_predictions.tsv'}")


if __name__ == "__main__":
    main()
