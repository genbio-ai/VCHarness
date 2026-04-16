"""Node 3-2-2-1-1: Frozen ESM2-3B + Frozen STRING_GNN Dual-Branch
             + Gated Fusion (FUSION_DIM=512) + 3-Block PreNorm MLP
             + Muon + CosineWarmRestarts + Manifold Mixup (prob=0.65)
             + Extended Training (patience=120, max_epochs=420)
=============================================================================
This node builds on the parent (node3-2-2-1, F1=0.5212, ESM2-650M+STRING)
with two targeted improvements:

1. UPGRADE ESM2-650M → ESM2-3B (primary):
   - ESM2-3B provides 2560-dim embeddings vs 1280-dim for ESM2-650M
   - The tree-best node (node3-1-1-1-1-2-1-1-1, F1=0.5283) uses ESM2-3B
   - Projected to FUSION_DIM=512 (same as ESM2-650M approach: 5:1 compression)
   - batch_size_esm reduced to 4 for ESM2-3B memory safety

2. EXTEND TRAINING to guarantee the epoch-240 CosineWarmRestarts restart (primary):
   - Parent stopped at epoch 231 (patience=80 from best at epoch 151)
   - The epoch-240 restart was 9 epochs away — historically worth +0.015–0.020 F1
   - patience: 80 → 120 (guarantees restart coverage: 151 + 120 = epoch 271 > 240)
   - max_epochs: 300 → 420 (allows cycle 3 to complete and explore)
   - This is the lowest-risk, highest-ROI improvement from the parent feedback

Architecture:
  - Input  : ENSG ID → two parallel frozen embedding branches
             a. Frozen STRING_GNN PPI embedding: 256-dim
             b. Frozen ESM2-3B protein sequence embedding: 2560-dim
             (precomputed via mean-pooling the last hidden state)
  - Fusion : Learnable sigmoidal gated fusion (FUSION_DIM=512)
             STRING branch: LN+Linear(256->512)
             ESM2 branch:   LN+Linear(2560->512)
             gate = sigmoid(Linear(1024->512))
             fused = gate * esm2_feat + (1-gate) * str_feat
  - Trunk  : input_proj(LN+Linear(512->384)+GELU+Dropout) →
             3x Pre-LayerNorm residual blocks (h=384, inner=768, dropout=0.30)
  - Head   : LayerNorm + Dropout(0.10) + Linear(384->6640*3) + per-gene bias

Training:
  - Loss      : weighted cross-entropy (class order: down=4.77%,
                neutral=92.82%, up=2.41%) - no focal loss, no label smoothing
  - Optimizer : Muon (lr=0.01) for 2D MLP trunk weights +
                AdamW (lr=3e-4, wd=8e-4) for projections/fusion/head/biases
  - Schedule  : CosineAnnealingWarmRestarts (T_0=80, T_mult=2)
                -> restarts at epochs 80, 240, 560 ...
  - Mixup     : Manifold Mixup at hidden space after input_proj (prob=0.65)
                soft-label KL divergence loss during mixup steps
  - Epochs    : max_epochs=420, early-stop patience=120

Key differences from parent (node3-2-2-1, ESM2-650M, F1=0.5212):
  * CHANGE: ESM2-650M (1280-dim) → ESM2-3B (2560-dim) — richer protein embeddings
  * CHANGE: ESM2 projection Linear(1280->512) → Linear(2560->512) (same 5:1 ratio)
  * CHANGE: patience: 80 → 120 (guarantees the epoch-240 restart is reached)
  * CHANGE: max_epochs: 300 → 420 (allows cycle 3 exploration: restart at 240+T0=560)
  * CHANGE: batch_size_esm: 8 → 4 (ESM2-3B requires more memory per batch)
  * ALL OTHER hyperparameters kept identical (FUSION_DIM=512, hidden=384, mixup_prob=0.65)

Evidence from tree:
  - node3-1-1-1-1-2-1-1-1 (ESM2-3B+STRING, FUSION_DIM=512): F1=0.5283 (tree-best)
  - node3-1-1-1-1-2-1-1   (ESM2-3B+STRING, FUSION_DIM=512): F1=0.5265
  - node3-2-2-1            (ESM2-650M+STRING, parent):       F1=0.5212
  ESM2-3B gap over ESM2-650M: +0.007 F1 in proven node3-1-1-1-1 lineage
  Patience/extended training gap: +0.015–0.020 F1 (based on cycle-1→2 improvement pattern)
  Combined expected target: F1=0.525–0.535
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
from transformers import AutoTokenizer, EsmForMaskedLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRING_GNN_DIR = "/home/Models/STRING_GNN"
PROTEIN_FASTA = "/home/data/genome/hg38_gencode_protein.fa"
ESM2_MODEL_NAME = "facebook/esm2_t36_3B_UR50D"   # ESM2-3B (tree-best)
ESM2_DIM = 2560      # ESM2-3B embedding dimension (vs 1280 for ESM2-650M)
GNN_DIM = 256        # STRING_GNN output embedding dimension
FUSION_DIM = 512     # Gated fusion dimension - proven 512 > 256 in tree
HIDDEN_DIM = 384     # MLP hidden dimension (Muon-compatible, proven optimal)
INNER_DIM = 768      # MLP inner (expansion) dimension (2x hidden)
N_GENES = 6640       # number of response genes per perturbation
N_CLASSES = 3        # down (-1->0), neutral (0->1), up (1->2)

# Fallback protein sequence for genes not found in FASTA
FALLBACK_SEQ = "MAAAAA"


# ---------------------------------------------------------------------------
# Protein sequence loading from FASTA
# ---------------------------------------------------------------------------
def _build_ensg_to_seq(fasta_path: str) -> Dict[str, str]:
    """Build ENSG->longest protein sequence map from GENCODE protein FASTA.

    Header format: >ENSP... gene:ENSG... transcript:...
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
                header = line[1:]
                current_ensg = None
                for part in header.split():
                    if part.startswith("gene:ENSG"):
                        current_ensg = part.split(":")[1]
                        break
            else:
                current_seq_parts.append(line)
    flush()

    # Keep the longest isoform per gene for maximum protein sequence coverage
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
# Model components
# ---------------------------------------------------------------------------
class PreNormResBlock(nn.Module):
    """Pre-LayerNorm residual block: LN -> Linear -> GELU -> Dropout -> Linear -> Dropout -> add."""

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
    """Learnable sigmoidal gated fusion of two FUSION_DIM feature vectors.

    Given STRING branch h_str (FUSION_DIM) and ESM2 branch h_esm (FUSION_DIM):
        gate = sigmoid(Linear(2*FUSION_DIM -> FUSION_DIM) applied to concat)
        fused = gate * h_esm + (1-gate) * h_str

    This allows the model to dynamically weight ESM2 and STRING contributions
    on a per-gene, per-dimension basis.
    """

    def __init__(self, in_dim: int, fusion_dim: int) -> None:
        super().__init__()
        self.gate_fc = nn.Linear(2 * in_dim, fusion_dim)

    def forward(self, h_str: torch.Tensor, h_esm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_str: [B, in_dim] - STRING branch projected features
            h_esm: [B, in_dim] - ESM2 branch projected features
        Returns:
            fused: [B, fusion_dim]
        """
        concat = torch.cat([h_str, h_esm], dim=-1)  # [B, 2*in_dim]
        gate = torch.sigmoid(self.gate_fc(concat))  # [B, fusion_dim]
        fused = gate * h_esm + (1.0 - gate) * h_str  # [B, fusion_dim]
        return fused


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        inner_dim: int = INNER_DIM,
        n_blocks: int = 3,
        dropout: float = 0.30,
        head_dropout: float = 0.10,
        adamw_lr: float = 3e-4,
        muon_lr: float = 0.01,
        weight_decay: float = 8e-4,
        cosine_t0: int = 80,
        cosine_t_mult: int = 2,
        min_lr: float = 1e-7,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.65,
        batch_size_esm: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Architecture modules (initialized in setup())
        self.str_proj: Optional[nn.Sequential] = None
        self.esm_proj: Optional[nn.Sequential] = None
        self.gate_fusion: Optional[GatedFusion] = None
        self.input_proj: Optional[nn.Sequential] = None
        self.blocks: Optional[nn.ModuleList] = None
        self.output_head: Optional[nn.Sequential] = None
        self.gene_bias: Optional[nn.Parameter] = None

        # Embedding lookup tables (populated in setup)
        self.gnn_id_to_idx: Dict[str, int] = {}
        self.esm2_id_to_idx: Dict[str, int] = {}

        # Accumulators for metrics
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

        # For ensemble
        self._final_test_ids: List[str] = []
        self._final_test_syms: List[str] = []
        self._final_test_preds: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        """Build model: precompute frozen STRING_GNN + ESM2-3B embeddings."""
        import torch.distributed as dist
        from transformers import AutoModel

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # ---- STRING_GNN frozen embeddings ----
        self.print("Loading STRING_GNN and computing frozen node embeddings ...")
        gnn_model = AutoModel.from_pretrained(
            STRING_GNN_DIR, trust_remote_code=True
        )
        gnn_model.eval()
        gnn_model = gnn_model.to(self.device)

        graph = torch.load(
            Path(STRING_GNN_DIR) / "graph_data.pt",
            map_location=self.device,
            weights_only=False,
        )
        edge_index = graph["edge_index"].to(self.device)
        edge_weight = graph.get("edge_weight")
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        with torch.no_grad():
            gnn_out = gnn_model(edge_index=edge_index, edge_weight=edge_weight)

        all_emb = gnn_out.last_hidden_state.detach().float()
        self.register_buffer("gnn_embeddings", all_emb)  # [18870, 256]

        del gnn_model, gnn_out, graph, edge_index, edge_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.print(f"STRING_GNN embeddings: {all_emb.shape}")

        # Build ENSG-ID -> row-index mapping
        node_names: List[str] = json.loads(
            (Path(STRING_GNN_DIR) / "node_names.json").read_text()
        )
        self.gnn_id_to_idx = {name: i for i, name in enumerate(node_names)}
        self.print(f"STRING_GNN covers {len(self.gnn_id_to_idx)} Ensembl gene IDs")

        # ---- ESM2-3B frozen embeddings ----
        self.print(f"Building ENSG -> protein sequence map from {PROTEIN_FASTA} ...")
        ensg2seq = _build_ensg_to_seq(PROTEIN_FASTA)
        self.print(f"FASTA contains {len(ensg2seq)} ENSG -> protein sequence entries")

        # Tokenizer: rank 0 downloads first
        self.print("Loading ESM2-3B tokenizer (rank 0 downloads first) ...")
        if local_rank == 0:
            AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)

        # ESM2 model: rank 0 downloads first
        self.print("Loading ESM2-3B model (rank 0 downloads first) ...")
        if local_rank == 0:
            EsmForMaskedLM.from_pretrained(ESM2_MODEL_NAME)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Load ESM2-3B in float32 for precomputation
        esm2_model = EsmForMaskedLM.from_pretrained(ESM2_MODEL_NAME, dtype=torch.float32)
        esm2_model.eval()
        esm2_model = esm2_model.to(self.device)

        # Collect all unique ENSG IDs: both FASTA-based and STRING GNN node names
        all_ensg_ids = sorted(ensg2seq.keys())
        all_ensg_ids_set: Set[str] = set(all_ensg_ids)
        for gnn_id in node_names:
            if gnn_id not in all_ensg_ids_set:
                all_ensg_ids.append(gnn_id)
                all_ensg_ids_set.add(gnn_id)

        self.print(
            f"Precomputing ESM2-3B embeddings for {len(all_ensg_ids)} ENSG IDs ..."
        )

        hp = self.hparams
        esm2_embeddings_list = []
        batch_size_esm = hp.batch_size_esm  # Small batch for ESM2-3B memory safety
        with torch.no_grad():
            for i in range(0, len(all_ensg_ids), batch_size_esm):
                batch_ids = all_ensg_ids[i : i + batch_size_esm]
                # Truncate to 1022 to leave room for cls/eos tokens
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
                hidden = outputs.hidden_states[-1]  # [B, seq_len, 2560] for ESM2-3B
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
                token_counts = (
                    (~special_mask).float().sum(dim=1, keepdim=True).clamp(min=1e-9)
                )
                mean_embeddings = masked_states.sum(dim=1) / token_counts  # [B, 2560]

                esm2_embeddings_list.append(mean_embeddings.detach().cpu().float())

                if (i // batch_size_esm) % 50 == 0:
                    self.print(
                        f"  ESM2-3B: processed {i + len(batch_ids)}/{len(all_ensg_ids)} proteins"
                    )

        # Register as non-trainable buffer [N_ensg, 2560]
        esm2_all_emb = torch.cat(esm2_embeddings_list, dim=0)  # [N_ensg, 2560]
        self.register_buffer("esm2_embeddings", esm2_all_emb)

        # Build ENSG-ID -> row-index mapping for ESM2 embeddings
        self.esm2_id_to_idx = {ensg: i for i, ensg in enumerate(all_ensg_ids)}
        self.print(f"ESM2-3B frozen embeddings: {esm2_all_emb.shape}")

        # Free ESM2 model memory after precomputation
        del esm2_model, esm2_embeddings_list, ensg2seq, tokenized, outputs, hidden
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- MLP architecture ----
        hp = self.hparams

        # STRING branch: LN + Linear(256 -> FUSION_DIM)
        self.str_proj = nn.Sequential(
            nn.LayerNorm(GNN_DIM),
            nn.Linear(GNN_DIM, FUSION_DIM),
        )

        # ESM2-3B branch: LN + Linear(2560 -> FUSION_DIM) — same 5:1 compression as proven tree-best
        self.esm_proj = nn.Sequential(
            nn.LayerNorm(ESM2_DIM),
            nn.Linear(ESM2_DIM, FUSION_DIM),
        )

        # Gated fusion: (FUSION_DIM, FUSION_DIM) -> FUSION_DIM
        self.gate_fusion = GatedFusion(in_dim=FUSION_DIM, fusion_dim=FUSION_DIM)

        # Input projection: FUSION_DIM -> hidden_dim (with activation and dropout)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Linear(FUSION_DIM, hp.hidden_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # 3x Pre-LayerNorm residual blocks
        self.blocks = nn.ModuleList(
            [
                PreNormResBlock(hp.hidden_dim, hp.inner_dim, hp.dropout)
                for _ in range(hp.n_blocks)
            ]
        )

        # Output head: LN + Dropout + Linear
        self.output_head = nn.Sequential(
            nn.LayerNorm(hp.hidden_dim),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.hidden_dim, N_GENES * N_CLASSES),
        )

        # Per-gene additive bias: one offset per (gene x class) pair
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES, N_CLASSES))

        # ---- Class weights ----
        # After label shift (-1->0, 0->1, 1->2):
        #   class 0 = down-regulated  (4.77%)  -> high weight
        #   class 1 = neutral         (92.82%) -> low weight
        #   class 2 = up-regulated    (2.41%)  -> highest weight
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
            f"Architecture: frozen ESM2-3B({ESM2_DIM})+STRING_GNN({GNN_DIM}) "
            f"-> GatedFusion({FUSION_DIM}) -> input_proj -> "
            f"{hp.n_blocks}xPreNormBlock({hp.hidden_dim},{hp.inner_dim}) "
            f"-> HeadDrop({hp.head_dropout}) -> Linear({hp.hidden_dim},{N_GENES}x{N_CLASSES})"
        )
        self.print(f"Trainable params: {trainable:,} / {total:,}")

    # ------------------------------------------------------------------
    def _get_str_embedding(self, pert_ids: List[str]) -> torch.Tensor:
        """Get frozen STRING_GNN embeddings for a batch of ENSG IDs."""
        str_list: List[torch.Tensor] = []
        for pid in pert_ids:
            row = self.gnn_id_to_idx.get(pid)
            if row is not None:
                str_list.append(self.gnn_embeddings[row])
            else:
                str_list.append(
                    torch.zeros(GNN_DIM, device=self.device, dtype=torch.float32)
                )
        return torch.stack(str_list, dim=0)  # [B, 256]

    def _get_esm2_embedding(self, pert_ids: List[str]) -> torch.Tensor:
        """Get frozen ESM2-3B embeddings from precomputed buffer."""
        esm_list: List[torch.Tensor] = []
        for pid in pert_ids:
            row = self.esm2_id_to_idx.get(pid)
            if row is not None:
                esm_list.append(self.esm2_embeddings[row])
            else:
                esm_list.append(
                    torch.zeros(ESM2_DIM, device=self.device, dtype=torch.float32)
                )
        return torch.stack(esm_list, dim=0)  # [B, 2560]

    def _get_fused(self, pert_ids: List[str]) -> torch.Tensor:
        """Get gated fused representation for a batch of ENSG IDs.

        Combines frozen STRING_GNN topology embeddings (256-dim) with
        frozen ESM2-3B protein sequence embeddings (2560-dim) via a
        learnable sigmoidal gate, both projected to FUSION_DIM=512.

        Returns:
            fused: [B, FUSION_DIM=512]
        """
        str_emb = self._get_str_embedding(pert_ids)    # [B, 256]
        esm_emb = self._get_esm2_embedding(pert_ids)   # [B, 2560]

        h_str = self.str_proj(str_emb)    # [B, 512]
        h_esm = self.esm_proj(esm_emb)    # [B, 512]
        fused = self.gate_fusion(h_str, h_esm)  # [B, 512]
        return fused

    def _forward_from_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual blocks and output head.

        Args:
            x: [B, hidden_dim] - features after input_proj
        Returns:
            logits: [B, N_CLASSES, N_GENES]
        """
        for block in self.blocks:
            x = block(x)                               # [B, hidden_dim]
        logits = self.output_head(x)                   # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)   # [B, 3, 6640]
        # gene_bias: [N_GENES, N_CLASSES].T -> [N_CLASSES, N_GENES] -> [1, 3, N_GENES]
        logits = logits + self.gene_bias.T.unsqueeze(0)
        return logits

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits of shape [B, N_CLASSES, N_GENES]. Used for val/test."""
        fused = self._get_fused(pert_ids)       # [B, FUSION_DIM]
        x = self.input_proj(fused)              # [B, hidden_dim]
        return self._forward_from_hidden(x)     # [B, 3, 6640]

    def _compute_loss_hard(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted CE on [B, N_CLASSES, N_GENES] logits with integer labels.

        No label smoothing: proven in tree that WCE without smoothing works
        better with Muon optimizer. Label smoothing conflicts with Muon's
        fast convergence.
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
        )

    def _compute_loss_soft(
        self, logits: torch.Tensor, mixed_labels_onehot: torch.Tensor
    ) -> torch.Tensor:
        """Soft-label KL divergence for Manifold Mixup training steps.

        Following the proven recipe from tree-best nodes (node3-3-1-2-1-1-1,
        node3-1-1-1-1-2-1-1-1): use KL divergence with class-weighted
        soft labels for more stable mixup learning.

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

        # Apply class weights (broadcast over batch*gene dimension)
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

        # Get gated fused embedding (frozen ESM2-3B + frozen STRING_GNN)
        fused = self._get_fused(pert_ids)  # [B, FUSION_DIM]
        x = self.input_proj(fused)         # [B, hidden_dim]

        hp = self.hparams
        use_mixup = self.training and (torch.rand(1).item() < hp.mixup_prob)

        if use_mixup:
            # Manifold Mixup in hidden space (after input_proj)
            batch_size = x.size(0)
            lam = float(
                torch.distributions.Beta(
                    torch.tensor(hp.mixup_alpha), torch.tensor(hp.mixup_alpha)
                ).sample()
            )
            lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5

            idx = torch.randperm(batch_size, device=x.device)
            mixed_x = lam * x + (1.0 - lam) * x[idx]

            # Convert integer labels to one-hot for soft-label loss
            labels_onehot = F.one_hot(labels, num_classes=N_CLASSES).float()  # [B, N_GENES, 3]
            mixed_labels_onehot = lam * labels_onehot + (1.0 - lam) * labels_onehot[idx]

            logits = self._forward_from_hidden(mixed_x)
            loss = self._compute_loss_soft(logits, mixed_labels_onehot)
        else:
            logits = self._forward_from_hidden(x)
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
            # All ranks computed the same global F1 via all_gather; sync_dist=True
            # averages identical values (no-op), but suppresses the DDP warning.
            self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        else:
            preds_np = preds_local.numpy()
            labels_np = labels_local.numpy()
            f1 = _compute_per_gene_f1(preds_np, labels_np)
            self.log("val_f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        self._test_preds.append(logits.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist

        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        self._test_preds.clear()

        # Gather tensor predictions from all ranks
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

        if self.trainer.is_global_zero:
            all_pert_ids = [pid for lst in gathered_pert_ids_flat for pid in lst]
            all_symbols = [sym for lst in gathered_symbols_flat for sym in lst]

            # De-duplicate (DDP may replicate samples across ranks)
            seen: Set[str] = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            preds_np = all_preds.cpu().numpy()  # [N_total, 3, 6640]
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

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams

        # Separate parameters: Muon for 2D hidden weight matrices in MLP trunk,
        # AdamW for everything else (projections, fusion, head, biases, norms)
        # Per Muon skill: Muon should NOT be applied to output layers, embeddings,
        # projection layers, or 1D parameters.
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
            f"Optimizer split: Muon={n_muon:,} params (MLP 2D block weights), "
            f"AdamW={n_adamw:,} params (projections + fusion + head + bias)"
        )

        try:
            from muon import MuonWithAuxAdam

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
            self.print("Using MuonWithAuxAdam optimizer")
        except ImportError:
            self.print("WARNING: Muon not available, falling back to AdamW")
            all_params = muon_params + adamw_params
            optimizer = torch.optim.AdamW(
                all_params,
                lr=hp.adamw_lr,
                weight_decay=hp.weight_decay,
            )

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
        """Save only trainable parameters and essential small buffers.

        With frozen ESM2-3B + frozen STRING_GNN, the large embedding buffers
        (gnn_embeddings ~18870x256, esm2_embeddings ~NxESM2_DIM) are excluded
        from checkpoint to avoid huge checkpoint files. They are reconstructed
        at setup() from the pretrained models.
        """
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        result = {}

        # Save trainable parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    result[key] = full_sd[key]

        # Save only small essential buffers (class_weights)
        # Exclude large frozen buffers (gnn_embeddings, esm2_embeddings)
        small_buffers = {"class_weights"}
        for name, buffer in self.named_buffers():
            if name in small_buffers:
                key = prefix + name
                if key in full_sd:
                    result[key] = full_sd[key]

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving checkpoint: {trainable:,}/{total:,} params "
            f"({100 * trainable / total:.2f}%), "
            f"total tensors={len(result)}"
        )
        return result

    def load_state_dict(self, state_dict, strict=True):
        """Load trainable params + small buffers from checkpoint (strict=False)."""
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute the per-gene macro-averaged F1 as defined in calc_metric.py.

    preds  : [N_samples, 3, N_genes]  — logits / probabilities
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
    print(f"Saved {len(rows)} test predictions -> {out_path}")


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------
def _ensemble_predict_from_checkpoints(
    trainer: pl.Trainer,
    model: PerturbModule,
    datamodule: PerturbDataModule,
    checkpoint_dir: Path,
    top_k: int = 5,
    threshold: float = 0.003,
    output_dir: Path = None,
) -> None:
    """Run test inference using top-K checkpoints and average predictions.

    Top-K selection is by val_f1 score parsed from checkpoint filename.
    Threshold filters out checkpoints more than `threshold` below the best.
    This is the proven ensemble strategy from node3-1-1-1-1-2-1-1-1 (F1=0.5283).

    Args:
        trainer: Lightning Trainer (used for test inference)
        model: the trained PerturbModule
        datamodule: the data module
        checkpoint_dir: directory containing .ckpt files
        top_k: maximum number of checkpoints to average
        threshold: only include checkpoints with val_f1 >= (best - threshold)
        output_dir: where to save ensemble predictions
    """
    import re

    if output_dir is None:
        output_dir = checkpoint_dir.parent

    # Find checkpoints and parse val_f1 scores
    ckpt_files = sorted(checkpoint_dir.glob("*.ckpt"))
    # Exclude last.ckpt from the ensemble candidates
    ckpt_files = [f for f in ckpt_files if "last" not in f.name]

    if not ckpt_files:
        print("No ensemble checkpoints found, skipping ensemble.")
        return

    # Parse F1 scores from filenames: e.g. "best-epoch=0123-val_f1=0.5241.ckpt"
    ckpt_with_scores = []
    for ckpt in ckpt_files:
        match = re.search(r"val[_/]f1[=_]([\d.]+)\.ckpt", ckpt.name)
        if match:
            score = float(match.group(1))
            ckpt_with_scores.append((score, ckpt))

    if not ckpt_with_scores:
        print("Could not parse val_f1 scores from checkpoint filenames, skipping ensemble.")
        return

    # Sort by descending score
    ckpt_with_scores.sort(key=lambda x: x[0], reverse=True)
    best_score = ckpt_with_scores[0][0]

    # Filter by threshold and limit to top_k
    selected = [
        (score, ckpt)
        for score, ckpt in ckpt_with_scores
        if score >= best_score - threshold
    ][:top_k]

    print(
        f"Ensemble: {len(selected)} checkpoints selected "
        f"(best={best_score:.4f}, threshold={threshold}, top_k={top_k})"
    )
    for score, ckpt in selected:
        print(f"  val_f1={score:.4f}: {ckpt.name}")

    if len(selected) < 2:
        print("Only 1 checkpoint for ensemble, no averaging needed.")
        return

    # Run test inference on each checkpoint and accumulate predictions
    all_preds_list = []
    for score, ckpt_path in selected:
        print(f"Running test inference with checkpoint: {ckpt_path.name}")
        trainer.test(model, datamodule=datamodule, ckpt_path=str(ckpt_path))

        if trainer.is_global_zero and model._final_test_preds is not None:
            all_preds_list.append(model._final_test_preds.copy())

    if trainer.is_global_zero and len(all_preds_list) > 1:
        # Average predictions across checkpoints
        ensemble_preds = np.mean(all_preds_list, axis=0)

        # Use IDs/symbols from the last run (all should be identical)
        pert_ids = model._final_test_ids
        symbols = model._final_test_syms

        ensemble_path = output_dir / "test_predictions.tsv"
        _save_test_predictions(
            pert_ids=pert_ids,
            symbols=symbols,
            preds=ensemble_preds,
            out_path=ensemble_path,
        )
        print(f"Ensemble predictions saved to {ensemble_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node3-2-2-1-1: Frozen ESM2-3B + STRING_GNN + GatedFusion + Muon + CosineWarmRestarts"
    )
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=420)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=8e-4)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--head-dropout", type=float, default=0.10)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--inner-dim", type=int, default=768)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--cosine-t0", type=int, default=80)
    p.add_argument("--cosine-t-mult", type=int, default=2)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--mixup-prob", type=float, default=0.65)
    p.add_argument("--early-stop-patience", type=int, default=120)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--batch-size-esm", type=int, default=4,
                   help="Batch size for ESM2-3B precomputation (smaller for larger models)")
    p.add_argument("--ensemble-top-k", type=int, default=5,
                   help="Number of top checkpoints to use for ensemble at test time")
    p.add_argument("--ensemble-threshold", type=float, default=0.003,
                   help="Only include checkpoints within this F1 of the best")
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
        adamw_lr=args.adamw_lr,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        cosine_t0=args.cosine_t0,
        cosine_t_mult=args.cosine_t_mult,
        min_lr=args.min_lr,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        batch_size_esm=args.batch_size_esm,
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

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=5,    # Save top 5 for ensemble
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
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval if (
            args.debug_max_step is None and not fast_dev_run
        ) else 1.0,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.fit(model, datamodule=datamodule)

    # ------------------------------------------------------------------
    # Test - use best checkpoint, then optionally ensemble
    # ------------------------------------------------------------------
    if args.fast_dev_run or args.debug_max_step is not None:
        # Debug mode: use current model (no checkpoint loading)
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        # Production mode: first test with best single checkpoint
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

        # Then run ensemble if we have multiple checkpoints
        # NOTE: All DDP ranks must enter this block together so that
        # trainer.test() inside _ensemble_predict_from_checkpoints can
        # coordinate across all processes (DDP requires all-rank participation).
        checkpoint_dir = Path(output_dir) / "checkpoints"
        if checkpoint_dir.exists():
            _ensemble_predict_from_checkpoints(
                trainer=trainer,
                model=model,
                datamodule=datamodule,
                checkpoint_dir=checkpoint_dir,
                top_k=args.ensemble_top_k,
                threshold=args.ensemble_threshold,
                output_dir=output_dir,
            )

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results saved -> {score_path}")


if __name__ == "__main__":
    main()
