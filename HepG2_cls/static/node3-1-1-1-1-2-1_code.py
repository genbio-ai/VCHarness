"""Node 3-1-1-1-1-2-1: Frozen ESM2-3B (2560-dim) + Frozen STRING_GNN Dual-Branch
              + Gated Fusion + 3-Block PreNorm MLP (h=384)
              + Muon Optimizer + Manifold Mixup
              + CosineAnnealingWarmRestarts + Top-7 Checkpoint Ensemble
================================================================
Parent  : node3-1-1-1-1-2 (Frozen ESM2-650M + STRING_GNN dual-branch, Test F1=0.5243)
          - Achieved tree-best F1=0.5243 with ESM2-650M (1280-dim) + STRING_GNN dual-branch
          - Near-zero val-test gap (+0.0003): excellent generalization
          - Feedback Priority 2: Try ESM2-3B (2560-dim) for richer protein representations
          - Feedback Priority 1: Expand ensemble from 5 to 7-10 checkpoints for diversity

Key Changes from Parent (node3-1-1-1-1-2)
-----------------------------------------------
1. UPGRADE ESM2-650M -> ESM2-3B (2560-dim):
   - ESM2-3B has 3B parameters vs 650M, providing significantly richer protein
     sequence representations (2560-dim vs 1280-dim)
   - Memory: ESM2-3B in bf16 ≈ 5.41 GB during precomputation (well within 80GB H100 VRAM)
   - After precompute: frozen buffer of 18870 × 2560 × 4 bytes ≈ 193 MB (fine for training)
   - Model is still frozen (not fine-tuned): no overfitting risk on 1,273 samples
   - ESM2 local path: facebook/esm2_t36_3B_UR50D (confirmed in HF cache)
   - This is the highest-priority improvement suggested by parent's feedback

2. EXPAND CHECKPOINT ENSEMBLE: Top-7 (from Top-5):
   - Ensemble from 5 → 7 best checkpoints
   - More diverse checkpoints across multiple cosine cycles capture complementary solutions
   - Expected +0.002-0.008 marginal gain in F1 from reduced prediction variance

3. ADJUST ESM2 BRANCH PROJECTION (2560→256):
   - LN(2560) + Linear(2560→256) replaces LN(1280) + Linear(1280→256)
   - Higher compression ratio (10:1 vs 5:1) but richer input: 2560-dim ESM2-3B
   - Gated fusion remains identical: concat([h_str, h_esm]) → sigmoid gate

4. PRECOMPUTE ESM2-3B in BFLOAT16 (memory efficiency):
   - Load ESM2-3B in bf16 during precomputation to fit in GPU memory
   - Store the precomputed buffer in float32 for stable downstream operations

All other architectural choices are preserved from the proven parent:
- Frozen STRING_GNN (256-dim), Gated sigmoid fusion, 3-block PreNorm MLP (h=384, inner=768)
- Muon (LR=0.01) for MLP 2D weights + AdamW (LR=3e-4) for others
- CosineAnnealingWarmRestarts (T_0=80, T_mult=2)
- Manifold Mixup (prob=0.65, alpha=0.2)
- Per-gene bias (N_GENES × N_CLASSES = 19,920 params)
- Label smoothing = 0.0, dropout=0.30, head_dropout=0.10

Architecture:
--------------------------------------------
Perturbed Gene (ENSG ID)
     │
     ├─── STRING_GNN frozen [18,870×256] → LN → Linear(256→256) → h_str
     │
     └─── ESM2-3B frozen (precomputed at setup in bf16, stored as float32 buffer)
              → mean_pool → [B, 2560] → LN → Linear(2560→256) → h_esm

     gate = sigmoid(Linear(512→256)([h_str, h_esm]))
     fused = gate * h_str + (1 - gate) * h_esm    [B, 256]

     LN(256) → Linear(256→384) → GELU → Dropout(0.30)   [input_proj]
          ↓ [Manifold Mixup applied here during training, prob=0.65]
     3×PreNormResBlock(h=384, inner=768, dropout=0.30)
     LN(384) → Dropout(0.10) → Linear(384→19,920) + gene_bias [N_GENES, N_CLASSES]

Training: Muon (LR=0.01) for MLP block 2D weight matrices +
          AdamW (LR=3e-4) for projections + fusion + head + gene_bias;
          CosineAnnealingWarmRestarts(T_0=80, T_mult=2, eta_min=1e-7);
          Weighted CE (no label smoothing); Manifold Mixup (alpha=0.2, prob=0.65).

Protein sequences sourced from: /home/data/genome/hg38_gencode_protein.fa
ESM2-3B model: facebook/esm2_t36_3B_UR50D (confirmed in HF cache)
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

# ESM2-3B: 3B parameters, 2560-dim embeddings (richer than ESM2-650M's 1280-dim)
ESM2_MODEL_NAME = "facebook/esm2_t36_3B_UR50D"

ESM2_DIM = 2560       # ESM2-3B hidden dimension (vs 1280 for ESM2-650M)
GNN_DIM = 256         # STRING_GNN output embedding dimension
FUSION_DIM = 256      # Gated fusion output dimension
HIDDEN_DIM = 384      # MLP hidden dimension
INNER_DIM = 768       # MLP inner (expansion) dimension (2x HIDDEN_DIM)
N_GENES = 6640        # Number of response genes per perturbation
N_CLASSES = 3         # down (-1→0), neutral (0→1), up (1→2)

# Fallback protein sequence for genes not found in FASTA
FALLBACK_SEQ = "MAAAAA"


# ---------------------------------------------------------------------------
# Protein sequence loading from GENCODE FASTA
# ---------------------------------------------------------------------------
def _build_ensg_to_seq(fasta_path: str) -> Dict[str, str]:
    """Build ENSG → longest protein sequence map from GENCODE protein FASTA.

    FASTA header format: >ENSP... gene:ENSG... transcript:...
    We parse the ENSG ID and keep only the longest isoform per gene.
    """
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
                header = line[1:]
                for part in header.split():
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
    """Each sample represents one gene perturbation experiment in HepG2 cells."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        if "label" in df.columns:
            # Shift labels from {-1, 0, 1} → {0, 1, 2} for CE loss
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
    """Pre-LayerNorm residual block (proven architecture in dual-modality nodes).

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
    """Learnable gated fusion of STRING and ESM2 branches.

    Given STRING branch h_str (FUSION_DIM) and ESM2 branch h_esm (FUSION_DIM):
        gate = sigmoid(Linear(2*FUSION_DIM → FUSION_DIM)([h_str, h_esm]))
        fused = gate * h_str + (1 - gate) * h_esm

    This allows the model to learn the optimal mixture of protein topology
    (STRING PPI) and protein sequence (ESM2) information for each gene.
    """

    def __init__(self, in_dim: int = FUSION_DIM) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(in_dim * 2, in_dim)

    def forward(self, h_str: torch.Tensor, h_esm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_str: [B, in_dim] — STRING branch projected features
            h_esm: [B, in_dim] — ESM2 branch projected features
        Returns:
            fused: [B, in_dim]
        """
        gate = torch.sigmoid(self.gate_proj(torch.cat([h_str, h_esm], dim=-1)))
        return gate * h_str + (1.0 - gate) * h_esm


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    """Frozen ESM2-3B + Frozen STRING_GNN Dual-Branch + GatedFusion + PreNorm MLP.

    Core design principles (validated in tree-best nodes):
    - Frozen ESM2-3B: precomputed protein sequence embeddings (2560-dim)
    - Frozen STRING_GNN: precomputed PPI topology embeddings (256-dim)
    - Gated fusion: learnable interpolation of both modalities
    - Manifold Mixup: strong regularization for small dataset (1,273 samples)
    - Muon optimizer: faster convergence for MLP block weight matrices
    - CosineWarmRestarts: escape local optima, extend training across cycles
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        inner_dim: int = INNER_DIM,
        n_blocks: int = 3,
        dropout: float = 0.30,
        head_dropout: float = 0.10,
        muon_lr: float = 0.01,
        adamw_lr: float = 3e-4,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.0,
        cosine_t0: int = 80,
        cosine_t_mult: int = 2,
        min_lr: float = 1e-7,
        grad_clip_norm: float = 2.0,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.65,
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

        # ENSG-ID → embedding row index mappings (populated in setup())
        self.gnn_id_to_idx: Dict[str, int] = {}
        self.esm2_id_to_idx: Dict[str, int] = {}

        # Metric accumulators
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        self._last_val_f1: float = float("nan")  # Store val/f1 for test phase proxy

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        """Build model: precompute frozen embeddings + construct MLP architecture."""
        import torch.distributed as dist
        from transformers import AutoTokenizer, EsmForMaskedLM
        from transformers import AutoModel

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # ---- STRING_GNN frozen embeddings ----
        self.print("Loading STRING_GNN and computing frozen node embeddings ...")
        gnn_model = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
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
        else:
            edge_weight = torch.ones(edge_index.shape[1], device=self.device)

        with torch.no_grad():
            gnn_out = gnn_model(edge_index=edge_index, edge_weight=edge_weight)

        # Register as non-trainable float32 buffer [18870, 256]
        all_gnn_emb = gnn_out.last_hidden_state.detach().float()
        self.register_buffer("gnn_embeddings", all_gnn_emb)

        # Free GNN model memory
        del gnn_model, gnn_out, graph, edge_index, edge_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.print(f"STRING_GNN embeddings: {all_gnn_emb.shape}")

        # Build ENSG-ID → row-index mapping for STRING_GNN
        node_names: List[str] = json.loads(
            (Path(STRING_GNN_DIR) / "node_names.json").read_text()
        )
        self.gnn_id_to_idx = {name: i for i, name in enumerate(node_names)}
        self.print(f"STRING_GNN covers {len(self.gnn_id_to_idx)} Ensembl gene IDs")

        # ---- Load protein FASTA for ESM2 inference ----
        self.print(f"Building ENSG → protein sequence map from {PROTEIN_FASTA} ...")
        ensg2seq = _build_ensg_to_seq(PROTEIN_FASTA)
        self.print(f"FASTA contains {len(ensg2seq)} ENSG → protein sequence entries")

        # ---- Precompute frozen ESM2-3B embeddings ----
        # ESM2-3B: 3B parameters, 2560-dim embeddings. Loaded in bf16 for memory efficiency.
        # Confirmed available at: facebook/esm2_t36_3B_UR50D (in HF cache)
        #
        # Memory budget analysis:
        # - ESM2-3B in bf16: ~5.41 GB GPU during precomputation
        # - After freeing: buffer of 18870 × 2560 × 4 bytes ≈ 193 MB (well within 80 GB H100 budget)
        self.print("Loading ESM2-3B tokenizer (rank 0 downloads first) ...")
        if local_rank == 0:
            AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)

        self.print("Loading ESM2-3B model in bf16 (rank 0 downloads first) ...")
        if local_rank == 0:
            EsmForMaskedLM.from_pretrained(ESM2_MODEL_NAME, torch_dtype=torch.bfloat16)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Load ESM2-3B in bf16 for memory-efficient precomputation
        esm2_model = EsmForMaskedLM.from_pretrained(
            ESM2_MODEL_NAME, torch_dtype=torch.bfloat16
        )
        esm2_model.eval()
        esm2_model = esm2_model.to(self.device)

        # Collect all unique ENSG IDs: FASTA + STRING_GNN node names
        all_ensg_ids = sorted(ensg2seq.keys())
        all_ensg_ids_set = set(all_ensg_ids)
        for gnn_id in node_names:
            if gnn_id not in all_ensg_ids_set:
                all_ensg_ids.append(gnn_id)
                all_ensg_ids_set.add(gnn_id)

        self.print(f"Precomputing ESM2-3B embeddings for {len(all_ensg_ids)} ENSG IDs ...")

        esm2_embeddings_list: List[torch.Tensor] = []
        # ESM2-3B is larger, use smaller batch size to avoid OOM
        batch_size_esm = 4  # Conservative for 3B model in bf16

        with torch.no_grad():
            for i in range(0, len(all_ensg_ids), batch_size_esm):
                batch_ids = all_ensg_ids[i:i + batch_size_esm]
                # Truncate sequences to max 1022 residues (leave 2 tokens for CLS/EOS)
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
                hidden = outputs.hidden_states[-1]  # [B, seq_len, 2560]
                special_ids_tensor = torch.tensor(
                    [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.eos_token_id],
                    device=self.device,
                )
                special_mask = torch.isin(tokenized["input_ids"], special_ids_tensor)
                masked_states = hidden.masked_fill(special_mask.unsqueeze(-1), 0.0)
                token_counts = (~special_mask).float().sum(dim=1, keepdim=True).clamp(min=1e-9)
                mean_embeddings = masked_states.sum(dim=1) / token_counts  # [B, 2560]

                # Store as float32 for stable downstream projection
                esm2_embeddings_list.append(mean_embeddings.detach().cpu().float())

                if (i // batch_size_esm) % 50 == 0:
                    self.print(
                        f"  ESM2-3B: processed {i + len(batch_ids)}/{len(all_ensg_ids)} proteins"
                    )

        # Register as non-trainable buffer [N_ensg, 2560]
        esm2_all_emb = torch.cat(esm2_embeddings_list, dim=0)
        self.register_buffer("esm2_embeddings", esm2_all_emb)
        self.esm2_id_to_idx = {ensg: i for i, ensg in enumerate(all_ensg_ids)}
        self.print(f"ESM2-3B frozen embeddings: {esm2_all_emb.shape}")

        # Free ESM2 model memory after precomputation
        del esm2_model, esm2_embeddings_list, ensg2seq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- MLP architecture ----
        hp = self.hparams

        # STRING branch: project GNN_DIM (256) → FUSION_DIM (256)
        self.str_proj = nn.Sequential(
            nn.LayerNorm(GNN_DIM),
            nn.Linear(GNN_DIM, FUSION_DIM),
        )

        # ESM2-3B branch: project ESM2_DIM (2560) → FUSION_DIM (256)
        # Higher compression ratio (10:1) vs ESM2-650M (5:1), but richer input
        self.esm_proj = nn.Sequential(
            nn.LayerNorm(ESM2_DIM),
            nn.Linear(ESM2_DIM, FUSION_DIM),
        )

        # Gated fusion: (FUSION_DIM, FUSION_DIM) → FUSION_DIM
        self.gate_fusion = GatedFusion(in_dim=FUSION_DIM)

        # Input projection: FUSION_DIM → hidden_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Linear(FUSION_DIM, hp.hidden_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # N PreNormResBlocks (h=384, inner=768) — proven optimal for dual-modality
        self.blocks = nn.ModuleList([
            PreNormResBlock(hp.hidden_dim, hp.inner_dim, hp.dropout)
            for _ in range(hp.n_blocks)
        ])

        # Output head with head dropout
        self.output_head = nn.Sequential(
            nn.LayerNorm(hp.hidden_dim),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.hidden_dim, N_GENES * N_CLASSES),
        )

        # Per-gene additive bias: [N_GENES, N_CLASSES] initialized to zero
        # Allows the model to learn gene-level prior class distributions
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES, N_CLASSES))

        # Class weights inversely proportional to frequency
        # class 0 = down-regulated (4.77%), class 1 = neutral (92.82%), class 2 = up-reg (2.41%)
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq) / (1.0 / freq).mean()
        self.register_buffer("class_weights", class_weights)

        # Cast all trainable parameters to float32 for stable optimization
        for k, v in self.named_parameters():
            if v.requires_grad:
                v.data = v.data.float()

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Architecture: frozen ESM2-3B({ESM2_DIM}) + STRING_GNN({GNN_DIM})"
            f" → GatedFusion({FUSION_DIM}) → Mixup → "
            f"{hp.n_blocks}×PreNormResBlock({hp.hidden_dim},{hp.inner_dim})"
            f" → HeadDropout({hp.head_dropout}) → Linear({hp.hidden_dim},{N_GENES}×{N_CLASSES}) + gene_bias"
        )
        self.print(f"Trainable params: {trainable:,} / {total:,}")

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------
    def _get_gnn_embedding(self, pert_ids: List[str]) -> torch.Tensor:
        """Get frozen STRING_GNN embeddings for a batch of ENSG IDs."""
        embs: List[torch.Tensor] = []
        for pid in pert_ids:
            row = self.gnn_id_to_idx.get(pid)
            if row is not None:
                embs.append(self.gnn_embeddings[row])
            else:
                embs.append(torch.zeros(GNN_DIM, device=self.device, dtype=torch.float32))
        return torch.stack(embs, dim=0)  # [B, 256]

    def _get_esm2_embedding(self, pert_ids: List[str]) -> torch.Tensor:
        """Get frozen ESM2-3B embeddings for a batch of ENSG IDs."""
        embs: List[torch.Tensor] = []
        for pid in pert_ids:
            row = self.esm2_id_to_idx.get(pid)
            if row is not None:
                embs.append(self.esm2_embeddings[row])
            else:
                embs.append(torch.zeros(ESM2_DIM, device=self.device, dtype=torch.float32))
        return torch.stack(embs, dim=0)  # [B, 2560]

    def _get_fused(self, pert_ids: List[str]) -> torch.Tensor:
        """Get gated fused representation for a batch of ENSG IDs.

        Combines frozen STRING_GNN PPI topology (256-dim) and frozen ESM2-3B
        protein sequence (2560-dim) via a learnable sigmoidal gate.

        Returns:
            fused: [B, FUSION_DIM=256]
        """
        str_emb = self._get_gnn_embedding(pert_ids)   # [B, 256]
        esm_emb = self._get_esm2_embedding(pert_ids)  # [B, 2560]

        h_str = self.str_proj(str_emb)   # [B, FUSION_DIM=256]
        h_esm = self.esm_proj(esm_emb)   # [B, FUSION_DIM=256]
        fused = self.gate_fusion(h_str, h_esm)  # [B, FUSION_DIM=256]
        return fused

    # ------------------------------------------------------------------
    # Manifold Mixup
    # ------------------------------------------------------------------
    def _manifold_mixup(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Apply Manifold Mixup: interpolate hidden representations and labels.

        Applied after input_proj (in the manifold space), before the residual blocks.
        Using Beta(alpha, alpha) distribution for the mixing coefficient.

        Args:
            x:      [B, hidden_dim] - features after input_proj
            labels: [B, N_GENES]    - integer class labels in {0, 1, 2}

        Returns:
            mixed_x:           [B, hidden_dim]        - mixed features
            mixed_labels_soft: [B, N_GENES, N_CLASSES] - soft mixed one-hot labels
            lam:               float                   - mixing coefficient
        """
        hp = self.hparams
        batch_size = x.size(0)

        # Sample mixing coefficient; ensure lam >= 0.5 for stable interpolation
        lam = float(
            torch.distributions.Beta(
                torch.tensor(hp.mixup_alpha),
                torch.tensor(hp.mixup_alpha),
            ).sample()
        )
        lam = max(lam, 1.0 - lam)

        # Random permutation within batch
        idx = torch.randperm(batch_size, device=x.device)

        # Mix features
        mixed_x = lam * x + (1.0 - lam) * x[idx]

        # Convert integer labels to one-hot and mix
        labels_oh = F.one_hot(labels, num_classes=N_CLASSES).float()  # [B, N_GENES, 3]
        mixed_labels_soft = lam * labels_oh + (1.0 - lam) * labels_oh[idx]

        return mixed_x, mixed_labels_soft, lam

    # ------------------------------------------------------------------
    # Forward pass helpers
    # ------------------------------------------------------------------
    def _forward_from_proj(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through blocks and output head (starting after input_proj)."""
        for block in self.blocks:
            x = block(x)                               # [B, hidden_dim]
        logits = self.output_head(x)                   # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)   # [B, 3, 6640]
        # gene_bias: [N_GENES, N_CLASSES].T → [N_CLASSES, N_GENES] → broadcast over B
        logits = logits + self.gene_bias.T.unsqueeze(0)
        return logits

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits [B, N_CLASSES, N_GENES]. Used for val/test."""
        fused = self._get_fused(pert_ids)   # [B, FUSION_DIM]
        x = self.input_proj(fused)          # [B, hidden_dim]
        return self._forward_from_proj(x)   # [B, 3, 6640]

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def _compute_loss_hard(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted cross-entropy with hard (integer) labels."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    def _compute_loss_soft(
        self, logits: torch.Tensor, mixed_labels_soft: torch.Tensor
    ) -> torch.Tensor:
        """KL-based soft cross-entropy for Manifold Mixup.

        Args:
            logits:            [B, N_CLASSES, N_GENES]
            mixed_labels_soft: [B, N_GENES, N_CLASSES] soft one-hot labels
        """
        logits_perm = logits.permute(0, 2, 1).float()   # [B, N_GENES, N_CLASSES]
        logits_flat = logits_perm.reshape(-1, N_CLASSES)  # [B*N_GENES, N_CLASSES]
        labels_flat = mixed_labels_soft.reshape(-1, N_CLASSES)  # [B*N_GENES, N_CLASSES]

        # Apply class weights: weight each class probability proportionally
        weighted_labels = labels_flat * self.class_weights.unsqueeze(0)
        weight_sum = weighted_labels.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weighted_labels = weighted_labels / weight_sum

        # Soft cross-entropy: -sum(soft_target * log_softmax(logit))
        log_probs = F.log_softmax(logits_flat, dim=-1)
        loss = -(weighted_labels * log_probs).sum(dim=-1).mean()
        return loss

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pert_ids = batch["pert_id"]
        labels = batch["label"]

        # Get gated fused embedding (frozen ESM2-3B + frozen STRING_GNN)
        fused = self._get_fused(pert_ids)  # [B, FUSION_DIM]
        x = self.input_proj(fused)         # [B, hidden_dim]

        hp = self.hparams
        # Apply Manifold Mixup with probability mixup_prob
        use_mixup = self.training and (torch.rand(1).item() < hp.mixup_prob)

        if use_mixup:
            mixed_x, mixed_labels_soft, _ = self._manifold_mixup(x, labels)
            logits = self._forward_from_proj(mixed_x)
            loss = self._compute_loss_soft(logits, mixed_labels_soft)
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
        else:
            preds_np = preds_local.numpy()
            labels_np = labels_local.numpy()

        f1 = _compute_per_gene_f1(preds_np, labels_np)
        self._last_val_f1 = f1  # Store for test phase proxy
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

        # Gather predictions from all DDP ranks
        gathered = self.all_gather(preds_local)        # [world_size, N_local, 3, 6640]
        all_preds = gathered.view(-1, N_CLASSES, N_GENES)  # [N_total, 3, 6640]

        # Gather string metadata from all ranks
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

        global_rank = dist.get_rank() if (is_dist and dist.is_initialized()) else 0
        if global_rank == 0:
            all_pert_ids = [pid for lst in gathered_pert_ids for pid in lst]
            all_symbols = [sym for lst in gathered_symbols for sym in lst]

            # Deduplicate by pert_id (handles DDP padding)
            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            preds_np = all_preds.cpu().numpy()
            for i, pid in enumerate(all_pert_ids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_ids.append(pid)
                    dedup_syms.append(all_symbols[i])
                    dedup_preds.append(preds_np[i])

            _save_test_predictions(
                pert_ids=dedup_ids,
                symbols=dedup_syms,
                preds=np.stack(dedup_preds, axis=0),
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

        # Log test/f1 for trainer.test() return value.
        # Test labels are not available in the model, so we log the val/f1 from
        # the last training epoch as a proxy. The real test metric is computed
        # by data/calc_metric.py on the test_predictions.tsv file (run by EvaluateAgent).
        proxy_f1 = self._last_val_f1
        self.log("test/f1", proxy_f1, on_epoch=True, prog_bar=True, sync_dist=True)

        # Also write test_score.txt directly from on_test_epoch_end to ensure
        # it is always written (not just when trainer.test() returns valid results).
        if global_rank == 0:
            score_path = Path(__file__).parent / "test_score.txt"
            score_path.write_text(str(proxy_f1))
            print(f"Test score → {score_path} (val/f1={proxy_f1})", flush=True)

    # ------------------------------------------------------------------
    # Optimizer and Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        from muon import MuonWithAuxAdam

        hp = self.hparams

        # Muon: 2D weight matrices in hidden residual MLP blocks
        # AdamW: all other trainable params (projections, fusion, head, bias, LN)
        muon_params = [
            p for name, p in self.blocks.named_parameters()
            if p.ndim >= 2 and p.requires_grad
        ]
        muon_param_ids = set(id(p) for p in muon_params)
        adamw_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in muon_param_ids
        ]

        n_muon = sum(p.numel() for p in muon_params)
        n_adamw = sum(p.numel() for p in adamw_params)
        self.print(
            f"Optimizer split: Muon={n_muon:,} (MLP 2D weights), "
            f"AdamW={n_adamw:,} (projections + fusion + head + bias)"
        )

        param_groups = [
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.weight_decay,
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

        # CosineAnnealingWarmRestarts: T_0=80, T_mult=2
        # Cycle lengths: 80 → 160 → 320 epochs
        # Proven in node3-3-1-2-1-1-1 (F1=0.5243): peaked at epoch 137 in cycle 2
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
    # Checkpoint state_dict: save only trainable params + small buffers
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and essential small buffers.

        The large frozen buffers (gnn_embeddings ~4.8MB, esm2_embeddings ~193MB)
        are excluded from checkpoints and reconstructed at setup(). Only class_weights
        buffer is included as a small essential buffer.
        """
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        trainable_sd = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    trainable_sd[key] = full_sd[key]

        # Include small essential buffers
        small_buffers = {"class_weights"}
        for name, _ in self.named_buffers():
            if name in small_buffers:
                key = prefix + name
                if key in full_sd:
                    trainable_sd[key] = full_sd[key]

        self.print(
            f"Saving checkpoint: {len(trainable_sd)} tensors, "
            f"~{sum(v.numel() for v in trainable_sd.values()):,} elements"
        )
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        """Load trainable params from checkpoint. strict=False handles frozen buffers."""
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-averaged F1, exactly matching data/calc_metric.py.

    Args:
        preds:  [N_samples, N_classes, N_genes] — logits/class scores
        labels: [N_samples, N_genes]            — integer class labels in {0,1,2}
    Returns:
        scalar F1 averaged over all genes (only present classes counted per gene)
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [N_samples, N_genes]
    n_genes = labels.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels[:, g]
        yh = y_hat[:, g]
        per_class_f1 = sk_f1(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        f1_vals.append(float(per_class_f1[present].mean()))
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    """Save test predictions as TSV matching the schema in data/test_predictions.tsv."""
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
    trainer: "pl.Trainer",
) -> None:
    """Load top-K checkpoints and average logits for final test predictions.

    With frozen ESM2-3B + frozen STRING_GNN, all checkpoints share the same
    precomputed embeddings stored in buffers. Loading each checkpoint only updates
    the MLP params (~10.6M), ensuring consistent and efficient ensemble inference.
    """
    if not checkpoint_paths:
        print("No checkpoints found for ensemble. Skipping.")
        return

    print(f"Running checkpoint ensemble with {len(checkpoint_paths)} checkpoints...")
    print("Checkpoints:", [Path(p).name for p in checkpoint_paths])

    if datamodule.test_ds is None:
        datamodule.setup("test")

    test_loader = datamodule.test_dataloader()

    # Unwrap DDP to get the underlying LightningModule for checkpoint loading
    # Use _lightning_module attribute (available in Lightning 2.5+)
    unwrapped_model: PerturbModule = trainer.strategy._lightning_module

    # Determine target device from the unwrapped model
    target_device = next(
        (p.device for p in unwrapped_model.parameters() if p.requires_grad),
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    # Move to target device once
    unwrapped_model = unwrapped_model.to(target_device)
    unwrapped_model.eval()

    all_ckpt_preds: List[np.ndarray] = []
    ensemble_ids: List[str] = []
    ensemble_syms: List[str] = []

    for idx, ckpt_path in enumerate(checkpoint_paths):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt)

            # Load into unwrapped model (strict=False handles frozen buffer keys not in ckpt)
            unwrapped_model.load_state_dict(sd, strict=False)

            ckpt_preds: List[torch.Tensor] = []
            ckpt_ids: List[str] = []
            ckpt_syms: List[str] = []

            with torch.no_grad():
                for batch in test_loader:
                    logits = unwrapped_model(batch["pert_id"])
                    ckpt_preds.append(logits.detach().cpu().float())
                    ckpt_ids.extend(batch["pert_id"])
                    ckpt_syms.extend(batch["symbol"])

            ckpt_preds_np = torch.cat(ckpt_preds, dim=0).numpy()
            all_ckpt_preds.append(ckpt_preds_np)

            # Collect metadata from the first successful checkpoint only
            # (all checkpoints predict the same test samples in the same order)
            if idx == 0:
                ensemble_ids = ckpt_ids
                ensemble_syms = ckpt_syms

            print(f"  Loaded: {Path(ckpt_path).name}, shape: {ckpt_preds_np.shape}")

        except Exception as exc:
            print(f"  Error loading checkpoint {ckpt_path}: {exc}. Skipping.")

    if not all_ckpt_preds:
        print("No valid checkpoints loaded. Ensemble aborted.")
        return

    # Average logits across all loaded checkpoints
    avg_preds = np.mean(all_ckpt_preds, axis=0)  # [N_test, 3, N_genes]

    # Deduplicate by pert_id (use metadata from first successful checkpoint)
    seen: set = set()
    dedup_ids, dedup_syms, dedup_preds = [], [], []
    for i, pid in enumerate(ensemble_ids):
        if pid not in seen:
            seen.add(pid)
            dedup_ids.append(pid)
            dedup_syms.append(ensemble_syms[i])
            dedup_preds.append(avg_preds[i])

    ensemble_preds = np.stack(dedup_preds, axis=0)

    ensemble_path = output_dir / "test_predictions.tsv"
    _save_test_predictions(
        pert_ids=dedup_ids,
        symbols=dedup_syms,
        preds=ensemble_preds,
        out_path=ensemble_path,
    )
    print(
        f"Ensemble predictions saved: {ensemble_path} "
        f"({len(dedup_ids)} samples, {len(all_ckpt_preds)} checkpoints)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Node3-1-1-1-1-2-1: Frozen ESM2-3B + STRING_GNN Dual-Branch + "
            "3-Block PreNorm MLP (h=384) + Muon + Mixup(prob=0.65) + "
            "CosineWarmRestarts + Top-7 Ensemble"
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
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Label smoothing (0.0 = no smoothing, proven best in dual-modality nodes)")
    p.add_argument("--dropout", type=float, default=0.30,
                   help="MLP trunk dropout")
    p.add_argument("--head-dropout", type=float, default=0.10,
                   help="Output head dropout (proven in node3-3-1-2-1-1-1)")
    p.add_argument("--hidden-dim", type=int, default=384,
                   help="MLP hidden dimension (proven optimal for this architecture)")
    p.add_argument("--inner-dim", type=int, default=768,
                   help="MLP inner expansion dimension (2x hidden_dim)")
    p.add_argument("--n-blocks", type=int, default=3,
                   help="Number of PreNormResBlocks (3 is optimal for dual-modality)")
    p.add_argument("--cosine-t0", type=int, default=80,
                   help="CosineWarmRestarts first cycle length")
    p.add_argument("--cosine-t-mult", type=int, default=2,
                   help="T_mult=2 for 80→160→320 cycles (proven in best dual-modality nodes)")
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--grad-clip-norm", type=float, default=2.0,
                   help="Gradient clipping max norm")
    p.add_argument("--mixup-alpha", type=float, default=0.2,
                   help="Beta distribution alpha for Manifold Mixup")
    p.add_argument("--mixup-prob", type=float, default=0.65,
                   help="Probability of applying Mixup per batch (proven in best nodes)")
    p.add_argument("--save-top-k", type=int, default=7,
                   help="Number of top checkpoints to save for ensemble (expanded to 7 for diversity)")
    p.add_argument("--early-stop-patience", type=int, default=80,
                   help="80 epochs = 1 full first cycle, allows recovery after warm restart")
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
    debug_max_step = args.debug_max_step

    if debug_max_step is not None:
        limit_train = debug_max_step * accumulate
        limit_val = limit_test = debug_max_step
        max_steps = debug_max_step
        val_check_interval = 1.0
        num_sanity_val_steps = 0
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1
        val_check_interval = args.val_check_interval if not fast_dev_run else 1.0
        num_sanity_val_steps = 2

    # Use auto_insert_metric_name=False to avoid Lightning creating subdirs for "/"
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-val_f1={val/f1:.4f}",
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
        min_delta=1e-5,
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
            find_unused_parameters=False,
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
            debug_max_step is None and not fast_dev_run
        ) else 1.0,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
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
    # Test: Best checkpoint, then ensemble if multiple checkpoints exist
    # ------------------------------------------------------------------
    if fast_dev_run or debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        # Run standard best-checkpoint test
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

        # Top-K checkpoint ensemble: average logits for improved predictions
        if trainer.is_global_zero:
            ckpt_dir = output_dir / "checkpoints"

            # Use rglob to find all .ckpt files recursively (avoids glob path bugs)
            ckpt_paths_raw = [
                p for p in ckpt_dir.rglob("*.ckpt")
                if "last" not in p.name
            ]

            # Sort by F1 value extracted from filename
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

                # Disable dropout for inference
                model.eval()
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = 0.0

                model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = model.to(model_device)

                datamodule.setup("test")
                _ensemble_test_predictions(
                    model=model,
                    datamodule=datamodule,
                    checkpoint_paths=ckpt_paths,
                    output_dir=output_dir,
                    trainer=trainer,
                )

    # ------------------------------------------------------------------
    # Save test score
    # ------------------------------------------------------------------
    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        if test_results and len(test_results) > 0:
            result = test_results[0]
            primary_metric = result.get("test/f1", result.get("test/f1_score", float("nan")))
        else:
            # Fallback: read from on_test_epoch_end's backup
            if score_path.exists():
                primary_metric = float(score_path.read_text().strip())
                print(f"Using fallback test score from {score_path}: {primary_metric}")
            else:
                primary_metric = float("nan")
        score_path.write_text(str(float(primary_metric)))
        print(f"Test results → {score_path} (f1_score={primary_metric})", flush=True)


if __name__ == "__main__":
    main()
