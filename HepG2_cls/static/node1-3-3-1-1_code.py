"""Node 1-3-3-1-1: ESM2-650M + STRING_GNN Dual-Branch
             + Gated Fusion + 3-Block PreNorm MLP (h=384)
             + Muon Optimizer + Manifold Mixup (prob=0.65)
             + CosineAnnealingWarmRestarts (T_0=80, T_mult=2)
             + Top-5 Checkpoint Ensemble

Parent : node1-3-3-1 (STRING-only, T_0=150, Mixup=0.3, F1=0.4930)
         - STRING-only architecture has hit a structural ceiling (~0.495 F1)
         - Frozen STRING embeddings provide no sample-adaptive signal
         - T_0=150 was counterproductive: only 3 restart cycles vs 6 for parent's T_0=80
         - Parent feedback Priority 1: "Abandon STRING-Only Lineage, Switch to ESM2+STRING"

Design Rationale:
-----------------
The parent feedback explicitly identifies the STRING-only architecture as having reached its
ceiling (~0.495 F1). The tree-best nodes (node3-1-1-1-1-2, node3-3-1-2-1-1-1, F1=0.5243)
all use frozen ESM2-650M + STRING_GNN dual-branch architecture, which proved that richer
feature representations with orthogonal biological signal (protein sequence + PPI topology)
are the key to breaking beyond the STRING-only ceiling.

Key changes from parent (node1-3-3-1):
1. ABANDON STRING-ONLY: Add frozen ESM2-650M (1280-dim) protein sequence embeddings
   alongside frozen STRING_GNN (256-dim) PPI topology embeddings.
   Rationale: Tree-best nodes (F1=0.5243) all use this dual-branch approach.
   Protein sequence evolutionary information is orthogonal to PPI graph topology.

2. GATED FUSION: Learnable sigmoidal gate fuses the two modalities.
   gate = sigmoid(Linear(512→256)([h_str, h_esm]))
   fused = gate * h_str + (1 - gate) * h_esm
   Rationale: Proven in node3-1-1-1-1-2 (F1=0.5243), node3-3-1-2-1-1-1 (F1=0.5243).
   Allows model to learn the optimal mixture of protein topology and sequence.

3. RESTORED T_0=80 (shorter warm restart cycles):
   Parent used T_0=150 which caused only 3 restarts in 500 epochs vs 6 for T_0=80.
   Feedback confirmed T_0=150 was counterproductive vs parent's T_0=80 (F1=0.4950 vs 0.4930).
   T_0=80, T_mult=2: restarts at epochs 80, 240, 560 — proven optimal for this architecture.

4. INCREASED MIXUP prob=0.30 → 0.65:
   Tree-best dual-modality nodes (F1=0.5243) used mixup_prob=0.65.
   With richer dual-modality features, stronger Mixup regularization is more beneficial.

5. REDUCED head_dropout=0.18 → 0.10:
   Tree-best dual-modality nodes used head_dropout=0.10.
   The dual-branch provides richer representations — less head dropout needed.

6. REDUCED trunk dropout=0.35 → 0.30:
   Tree-best dual-modality nodes used dropout=0.30 in residual blocks.

7. TOP-5 CHECKPOINT ENSEMBLE: Robust rglob-based checkpoint discovery (avoids glob bugs).

Architecture:
--------------------------------------------
Perturbed Gene (ENSG ID)
     │
     ├─── STRING_GNN frozen [18,870×256] → LN(256) → Linear(256→256) → h_str  [B, 256]
     │
     └─── ESM2-650M frozen (precomputed at setup)
              → mean_pool → [B, 1280] → LN(1280) → Linear(1280→256) → h_esm  [B, 256]

     gate = sigmoid(Linear(512→256)([h_str, h_esm]))
     fused = gate * h_str + (1 - gate) * h_esm              [B, 256]
             │
     ┌──────────────────────────────────────────┐
     │  3× PreNorm Residual Block (h=384)        │
     │  LN(256) → Linear(256→384) [input proj]   │
     │  3× LN(384) → Linear(384→768) → GELU →   │
     │      Dropout(0.30) → Linear(768→384) →    │
     │      Dropout(0.30) + residual GELU        │
     └──────────────────────────────────────────┘
             │ [B, 384]
     head_norm(LN) → head_dropout(0.10) → Linear(384→19920) + gene_bias [B, 19920]
             ↓ reshape
     [B, 3, 6640]
             │
     (test) Top-5 checkpoint ensemble: average logits → argmax → predictions

Training:
  Muon(LR=0.01) for 2D weight matrices in residual blocks
  AdamW(LR=3e-4) for projections, norms, biases, gene_bias, fusion
  CosineAnnealingWarmRestarts(T_0=80, T_mult=2, eta_min=1e-6)
  Manifold Mixup(prob=0.65, alpha=0.2) in hidden [B, 256] representation space
  Weighted cross-entropy (no label smoothing)
  Gradient clipping max_norm=1.0

References:
  - node3-1-1-1-1-2 (F1=0.5243, tree-best tied): frozen ESM2-650M + STRING gated, Mixup 0.65
  - node3-3-1-2-1-1-1 (F1=0.5243, tree-best): same architecture confirming recipe
  - node1-3-3 (parent, F1=0.4950): STRING-only with T_0=80 outperformed T_0=150 child
  - parent feedback: Priority 1 recommends ESM2+STRING dual-branch
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import glob as glob_module
import json
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640
N_CLASSES = 3
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ESM2_DIM = 1280       # ESM2-650M hidden dimension
GNN_DIM = 256         # STRING_GNN output embedding dimension
FUSION_DIM = 256      # Common fusion dimension for both branches
PROTEIN_FASTA = "/home/data/genome/hg38_gencode_protein.fa"

# Fallback protein sequence for genes not found in FASTA
FALLBACK_SEQ = "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLSAESVGEVYIKSTETGQYLAMDTSGLLYGSQTPSEECLFLERLEENHEGKQSELVHKLAKVNRELPPHLKKVLAEQEQAPSTARLQEAAQKMQRALLEERDRQLRGSS"


# ---------------------------------------------------------------------------
# Protein sequence loading from GENCODE FASTA
# ---------------------------------------------------------------------------
def _build_ensg_to_seq(fasta_path: str) -> Dict[str, str]:
    """Build ENSG → longest protein sequence map from GENCODE protein FASTA.

    FASTA header format: >ENSP... gene:ENSG... transcript:...
    We select the longest protein sequence per ENSG ID.
    """
    ensg2seq: Dict[str, str] = {}
    current_ensg: Optional[str] = None
    current_seq_parts: List[str] = []

    def flush() -> None:
        if current_ensg is None:
            return
        seq = "".join(current_seq_parts)
        if len(seq) > len(ensg2seq.get(current_ensg, "")):
            ensg2seq[current_ensg] = seq

    with open(fasta_path, "r") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                flush()
                current_seq_parts = []
                current_ensg = None
                for part in line.split():
                    if part.startswith("gene:"):
                        raw = part[5:]
                        # Strip version number (e.g., ENSG000001.5 → ENSG000001)
                        current_ensg = raw.split(".")[0]
                        break
            else:
                current_seq_parts.append(line)
    flush()
    return ensg2seq


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Gene-perturbation → differential-expression dataset.

    Stores only pert_ids/symbols/labels; embeddings are retrieved via index lookup
    during forward pass.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        if "label" in df.columns:
            labels = np.array([json.loads(x) for x in df["label"]], dtype=np.int64)
            self.labels: Optional[torch.Tensor] = torch.tensor(labels + 1, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
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
        micro_batch_size: int = 16,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.train_ds = self.val_ds = self.test_ds = None

    def setup(self, stage: str = "fit") -> None:
        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbDataset(train_df)
        self.val_ds = PerturbDataset(val_df)
        self.test_ds = PerturbDataset(test_df)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.micro_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------
class PreNormResBlock(nn.Module):
    """Pre-LayerNorm residual MLP block (h → 2h → h)."""

    def __init__(self, dim: int, inner_dim: int, dropout: float = 0.30) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(self.norm(x)))


class GatedFusion(nn.Module):
    """Learnable gated fusion of STRING and ESM2 branches.

    Given STRING branch h_str (FUSION_DIM) and ESM2 branch h_esm (FUSION_DIM):
        gate = sigmoid(Linear(2*FUSION_DIM → FUSION_DIM)([h_str, h_esm]))
        fused = gate * h_str + (1 - gate) * h_esm

    The gate is per-dimension and per-sample, learning to weight the relative
    contribution of PPI topology (STRING) vs protein sequence (ESM2) signal.
    """

    def __init__(self, in_dim: int = FUSION_DIM) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(in_dim * 2, in_dim)

    def forward(self, h_str: torch.Tensor, h_esm: torch.Tensor) -> torch.Tensor:
        """
        h_str: [B, in_dim] — STRING branch projected features
        h_esm: [B, in_dim] — ESM2 branch projected features
        Returns: fused [B, in_dim]
        """
        gate = torch.sigmoid(self.gate_proj(torch.cat([h_str, h_esm], dim=-1)))
        return gate * h_str + (1.0 - gate) * h_esm


class DualBranchModel(nn.Module):
    """Frozen ESM2-650M + Frozen STRING_GNN Dual-Branch with Gated Fusion.

    Architecture:
      STRING_GNN embed [B, 256] → LN+proj → h_str [B, 256]
      ESM2-650M embed  [B, 1280] → LN+proj → h_esm [B, 256]
      GatedFusion → fused [B, 256]
      Input projection [256→384] → 3× PreNormResBlock(384) →
      head_norm → head_dropout → Linear(384→19920) + gene_bias →
      Reshape [B, 3, 6640]
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.30,
        head_dropout: float = 0.10,
    ) -> None:
        super().__init__()

        # STRING branch: LN + projection 256 → FUSION_DIM
        self.str_norm = nn.LayerNorm(GNN_DIM)
        self.str_proj = nn.Linear(GNN_DIM, FUSION_DIM)

        # ESM2 branch: LN + projection 1280 → FUSION_DIM
        self.esm_norm = nn.LayerNorm(ESM2_DIM)
        self.esm_proj = nn.Linear(ESM2_DIM, FUSION_DIM)

        # Gated fusion
        self.fusion = GatedFusion(in_dim=FUSION_DIM)

        # Learnable fallback embedding for genes not in STRING graph (~6%)
        self.fallback_str_emb = nn.Parameter(torch.zeros(GNN_DIM))
        nn.init.normal_(self.fallback_str_emb, std=0.02)

        # Learnable fallback for ESM2 embeddings (genes not in FASTA)
        self.fallback_esm_emb = nn.Parameter(torch.zeros(ESM2_DIM))
        nn.init.normal_(self.fallback_esm_emb, std=0.02)

        # Input projection: FUSION_DIM → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(FUSION_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Residual MLP blocks (PreLN)
        self.blocks = nn.ModuleList(
            [PreNormResBlock(hidden_dim, hidden_dim * 2, dropout) for _ in range(n_blocks)]
        )

        # Output head
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head_dropout = nn.Dropout(head_dropout)
        self.head_linear = nn.Linear(hidden_dim, N_GENES * N_CLASSES)

        # Per-gene additive bias: captures baseline DE priors per response gene
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES * N_CLASSES))

    def forward(
        self,
        pert_ids: List[str],
        str_idx: torch.Tensor,       # [B]  STRING node indices, -1 = not in graph
        string_embs: torch.Tensor,   # [18870, 256] frozen buffer
        esm2_embs: torch.Tensor,     # [N_all_genes, 1280] frozen buffer (all seen genes)
        esm2_idx: torch.Tensor,      # [B]  ESM2 buffer indices, -1 = not available
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, hidden) where hidden is the pre-head fused representation."""
        batch_size = str_idx.shape[0]

        # ---- STRING branch ----
        valid_str = str_idx >= 0
        safe_str_idx = str_idx.clamp(min=0)
        h_str_raw = string_embs[safe_str_idx].to(torch.float32)  # [B, 256]
        if not valid_str.all():
            fallback_s = self.fallback_str_emb.to(h_str_raw).unsqueeze(0).expand(
                int((~valid_str).sum()), -1
            )
            h_str_raw = h_str_raw.clone()
            h_str_raw[~valid_str] = fallback_s
        h_str = self.str_proj(self.str_norm(h_str_raw))  # [B, FUSION_DIM]

        # ---- ESM2 branch ----
        valid_esm = esm2_idx >= 0
        safe_esm_idx = esm2_idx.clamp(min=0)
        h_esm_raw = esm2_embs[safe_esm_idx].to(torch.float32)  # [B, 1280]
        if not valid_esm.all():
            fallback_e = self.fallback_esm_emb.to(h_esm_raw).unsqueeze(0).expand(
                int((~valid_esm).sum()), -1
            )
            h_esm_raw = h_esm_raw.clone()
            h_esm_raw[~valid_esm] = fallback_e
        h_esm = self.esm_proj(self.esm_norm(h_esm_raw))  # [B, FUSION_DIM]

        # ---- Gated Fusion ----
        fused = self.fusion(h_str, h_esm)  # [B, FUSION_DIM=256]

        # ---- MLP ----
        x = self.input_proj(fused)    # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)

        # ---- Output head ----
        h = self.head_norm(x)
        h = self.head_dropout(h)
        logits = self.head_linear(h) + self.gene_bias.to(h)
        return logits.view(-1, N_CLASSES, N_GENES), x  # ([B, 3, 6640], [B, hidden_dim])


# ---------------------------------------------------------------------------
# Manifold Mixup helper
# ---------------------------------------------------------------------------
def manifold_mixup(
    x: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Manifold Mixup in the hidden representation space."""
    batch_size = x.shape[0]
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, labels, labels[index], lam


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.30,
        head_dropout: float = 0.10,
        muon_lr: float = 0.01,
        adamw_lr: float = 3e-4,
        weight_decay: float = 8e-4,
        t0: int = 80,
        t_mult: int = 2,
        mixup_prob: float = 0.65,
        mixup_alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.muon_lr = muon_lr
        self.adamw_lr = adamw_lr
        self.weight_decay = weight_decay
        self.t0 = t0
        self.t_mult = t_mult
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha

        self.model: Optional[DualBranchModel] = None
        self.ensg2str_idx: Dict[str, int] = {}
        self.ensg2esm_idx: Dict[str, int] = {}
        self.all_ensg_ids: List[str] = []

        # Accumulation buffers for epoch-level metrics
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # Class weights: inversely proportional to class frequencies
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq)
        class_weights = class_weights / class_weights.sum() * N_CLASSES
        self.register_buffer("class_weights", class_weights)

        if self.model is not None:
            return  # Already initialized

        # ---- Load STRING_GNN node embeddings ----
        from transformers import AutoModel
        string_gnn_dir = Path(str(STRING_GNN_DIR))
        gnn = AutoModel.from_pretrained(str(string_gnn_dir), trust_remote_code=True)
        gnn.eval()
        graph = torch.load(
            string_gnn_dir / "graph_data.pt", map_location="cpu", weights_only=False
        )
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)
        with torch.no_grad():
            gnn_out = gnn(edge_index=edge_index, edge_weight=edge_weight)
        string_embs = gnn_out.last_hidden_state.detach().float().cpu()  # [18870, 256]
        del gnn, gnn_out

        # Build ENSEMBL ID → STRING node index mapping
        node_names: List[str] = json.loads(
            (string_gnn_dir / "node_names.json").read_text()
        )
        self.ensg2str_idx = {ensg: i for i, ensg in enumerate(node_names)}

        # Register as frozen buffer
        self.register_buffer("string_embs", string_embs)

        # ---- Load ESM2-650M protein embeddings ----
        self.print(f"Building ENSG → protein sequence map from {PROTEIN_FASTA} ...")
        ensg2seq = _build_ensg_to_seq(PROTEIN_FASTA)
        self.print(f"FASTA contains {len(ensg2seq)} ENSG → protein sequence entries")

        # Collect all unique ENSG IDs from STRING + train/val/test data
        # Use all STRING node names as the universe
        self.all_ensg_ids = list(node_names)  # 18870 proteins

        from transformers import AutoTokenizer, AutoModel

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(ESM2_MODEL_NAME, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        esm2_tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME, trust_remote_code=True)
        # Use AutoModel to get hidden states (EsmForMaskedLM returns MaskedLMOutput tuple, not dict)
        esm2_model = AutoModel.from_pretrained(
            ESM2_MODEL_NAME, trust_remote_code=True, dtype=torch.bfloat16
        )
        esm2_model.eval()
        # Use GPU for faster embedding computation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        esm2_model = esm2_model.to(device)

        self.print(f"Computing ESM2 embeddings for {len(self.all_ensg_ids)} proteins on {device}...")

        esm2_embeddings_list: List[torch.Tensor] = []
        # Use larger batch size for GPU efficiency
        batch_size_esm = 64 if device.type == "cuda" else 8
        special_token_ids = torch.tensor(
            [esm2_tokenizer.pad_token_id, esm2_tokenizer.cls_token_id,
             esm2_tokenizer.eos_token_id], device=device
        )

        with torch.no_grad():
            for i in range(0, len(self.all_ensg_ids), batch_size_esm):
                batch_ids = self.all_ensg_ids[i: i + batch_size_esm]
                seqs = []
                for ensg in batch_ids:
                    seq = ensg2seq.get(ensg, FALLBACK_SEQ)
                    # Truncate to 1022 residues (leave 2 tokens for CLS/EOS)
                    seqs.append(seq[:1022])

                encoded = esm2_tokenizer(
                    seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = esm2_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                token_reps = out["hidden_states"][33].float()  # [B, seq_len, 1280]
                mask = torch.isin(input_ids, special_token_ids)
                masked_reps = token_reps.masked_fill(mask.unsqueeze(-1), 0.0)
                count = (~mask).float().sum(dim=1, keepdim=True).clamp(min=1e-9)
                mean_embeddings = masked_reps.sum(dim=1) / count  # [B, 1280]
                esm2_embeddings_list.append(mean_embeddings.detach().cpu().float())

                if (i // batch_size_esm) % 50 == 0:
                    self.print(
                        f"  ESM2: processed {min(i + batch_size_esm, len(self.all_ensg_ids))}/{len(self.all_ensg_ids)} proteins"
                    )

        esm2_all_emb = torch.cat(esm2_embeddings_list, dim=0)  # [N_all, 1280]
        self.register_buffer("esm2_embeddings", esm2_all_emb)
        self.print(f"ESM2 embeddings shape: {esm2_all_emb.shape}")

        # Build ENSEMBL ID → ESM2 buffer index mapping
        self.ensg2esm_idx = {ensg: i for i, ensg in enumerate(self.all_ensg_ids)}

        del esm2_model, esm2_embeddings_list, ensg2seq
        torch.cuda.empty_cache()

        # ---- Build model ----
        self.model = DualBranchModel(
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
            head_dropout=self.head_dropout,
        )

        # Cast trainable parameters to float32 for stable optimization
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        self.print(
            f"Node1-3-3-1-1 DualBranchModel | hidden={self.hidden_dim} | "
            f"blocks={self.n_blocks} | dropout={self.dropout} | "
            f"head_dropout={self.head_dropout} | muon_lr={self.muon_lr} | "
            f"t0={self.t0}/t_mult={self.t_mult} | "
            f"mixup_prob={self.mixup_prob}/alpha={self.mixup_alpha} | "
            f"weight_decay={self.weight_decay} | "
            f"trainable={n_trainable:,}/{n_total:,}"
        )

    def _get_str_idx(self, pert_ids: List[str]) -> torch.Tensor:
        """Convert pert_ids to STRING node indices."""
        return torch.tensor(
            [self.ensg2str_idx.get(pid, -1) for pid in pert_ids],
            dtype=torch.long
        )

    def _get_esm_idx(self, pert_ids: List[str]) -> torch.Tensor:
        """Convert pert_ids to ESM2 buffer indices."""
        return torch.tensor(
            [self.ensg2esm_idx.get(pid, -1) for pid in pert_ids],
            dtype=torch.long
        )

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted cross-entropy loss (no label smoothing)."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=0.0,
        )

    def _compute_mixed_loss(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Mixed cross-entropy loss for Manifold Mixup training."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)
        la_flat = labels_a.reshape(-1)
        lb_flat = labels_b.reshape(-1)
        loss_a = F.cross_entropy(logits_flat, la_flat, weight=self.class_weights, label_smoothing=0.0)
        loss_b = F.cross_entropy(logits_flat, lb_flat, weight=self.class_weights, label_smoothing=0.0)
        return lam * loss_a + (1 - lam) * loss_b

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pert_ids = batch["pert_id"]
        labels = batch["label"]

        # Get indices on the correct device
        str_idx = self._get_str_idx(pert_ids).to(self.device)
        esm_idx = self._get_esm_idx(pert_ids).to(self.device)

        # Full forward pass to get hidden representation
        logits, x = self.model(
            pert_ids, str_idx, self.string_embs, self.esm2_embeddings, esm_idx
        )

        # Apply Manifold Mixup in the fused hidden space (after fusion, before MLP head)
        apply_mixup = (
            self.training
            and np.random.random() < self.mixup_prob
            and labels is not None
        )

        if apply_mixup:
            mixed_x, labels_a, labels_b, lam = manifold_mixup(x, labels, alpha=self.mixup_alpha)

            # Forward mixed representation through output head only
            h = self.model.head_norm(mixed_x)
            h = self.model.head_dropout(h)
            logits_mixed = self.model.head_linear(h) + self.model.gene_bias.to(h)
            logits_mixed = logits_mixed.view(-1, N_CLASSES, N_GENES)

            loss = self._compute_mixed_loss(logits_mixed, labels_a, labels_b, lam)
        else:
            loss = self._compute_loss(logits, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        pert_ids = batch["pert_id"]
        str_idx = self._get_str_idx(pert_ids).to(self.device)
        esm_idx = self._get_esm_idx(pert_ids).to(self.device)
        logits, _ = self.model(
            pert_ids, str_idx, self.string_embs, self.esm2_embeddings, esm_idx
        )
        loss = self._compute_loss(logits, batch["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_preds.append(logits.detach().cpu())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds_local = torch.cat(self._val_preds, dim=0)    # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0)  # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        # Gather across DDP ranks for accurate global F1
        all_preds = self.all_gather(preds_local)   # [world_size, N_local, 3, 6640]
        all_labels = self.all_gather(labels_local) # [world_size, N_local, 6640]
        ws = self.trainer.world_size
        if ws > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
            all_labels = all_labels.view(-1, N_GENES)
        else:
            if all_preds.dim() == 4:
                all_preds = all_preds.squeeze(0)
                all_labels = all_labels.squeeze(0)

        preds_np = all_preds.float().cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        f1 = _compute_per_gene_f1(preds_np, labels_np)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        pert_ids = batch["pert_id"]
        str_idx = self._get_str_idx(pert_ids).to(self.device)
        esm_idx = self._get_esm_idx(pert_ids).to(self.device)
        logits, _ = self.model(
            pert_ids, str_idx, self.string_embs, self.esm2_embeddings, esm_idx
        )
        self._test_preds.append(logits.detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        self._test_preds.clear()

        all_preds = self.all_gather(preds_local)  # [world_size, N_local, 3, 6640]
        ws = self.trainer.world_size
        if ws > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
        else:
            if all_preds.dim() == 4:
                all_preds = all_preds.squeeze(0)

        # Gather string metadata from all ranks → rank 0 only
        if ws > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            if self.trainer.is_global_zero:
                _pert_gathered: List[List[str]] = [[] for _ in range(ws)]
                _syms_gathered: List[List[str]] = [[] for _ in range(ws)]
                torch.distributed.gather_object(self._test_pert_ids, _pert_gathered, dst=0)
                torch.distributed.gather_object(self._test_symbols, _syms_gathered, dst=0)
                all_pert_ids: List[str] = []
                all_symbols: List[str] = []
                for p_list, s_list in zip(_pert_gathered, _syms_gathered):
                    all_pert_ids.extend(p_list)
                    all_symbols.extend(s_list)
            else:
                torch.distributed.gather_object(self._test_pert_ids, dst=0)
                torch.distributed.gather_object(self._test_symbols, dst=0)
                all_pert_ids, all_symbols = [], []
        else:
            all_pert_ids = self._test_pert_ids
            all_symbols = self._test_symbols

        if self.trainer.is_global_zero:
            preds_np = all_preds.float().cpu().numpy()  # [N_total, 3, 6640]
            _save_test_predictions(
                pert_ids=all_pert_ids,
                symbols=all_symbols,
                preds=preds_np,
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        """Configure Muon+AdamW optimizer with CosineAnnealingWarmRestarts.

        Muon is applied to hidden 2D weight matrices in residual blocks.
        AdamW is applied to all other parameters.

        CosineAnnealingWarmRestarts with T_0=80/T_mult=2:
        - Restarts at epochs 80, 240, 560, ...
        - Proven optimal schedule from tree-best nodes (F1=0.5243)
        - Restored from parent's T_0=150 which was counterproductive (fewer restart cycles)
        """
        try:
            from muon import MuonWithAuxAdam
            use_muon = True
        except ImportError:
            use_muon = False
            self.print("Muon not installed — falling back to AdamW for all parameters")

        if use_muon:
            muon_params = []
            adamw_params = []

            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                # Apply Muon only to 2D weight matrices inside residual blocks
                is_block_weight = (
                    "model.blocks" in name
                    and "net" in name
                    and ("net.0.weight" in name or "net.3.weight" in name)
                    and param.ndim >= 2
                )
                if is_block_weight:
                    muon_params.append(param)
                else:
                    adamw_params.append(param)

            param_groups = [
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=self.muon_lr,
                    momentum=0.95,
                    weight_decay=self.weight_decay,
                ),
                dict(
                    params=adamw_params,
                    use_muon=False,
                    lr=self.adamw_lr,
                    betas=(0.9, 0.95),
                    weight_decay=self.weight_decay,
                ),
            ]

            optimizer = MuonWithAuxAdam(param_groups)
            self.print(
                f"Using Muon+AdamW: {len(muon_params)} Muon params, "
                f"{len(adamw_params)} AdamW params"
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.adamw_lr, weight_decay=self.weight_decay
            )

        # CosineAnnealingWarmRestarts — proven optimal for dual-modality architecture
        # T_0=80 restored (vs parent's T_0=150): more restart cycles = more exploration
        # Tree-best nodes (F1=0.5243) peaked at epoch 137 (cycle 2 with T_0=80, T_mult=2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.t0,
            T_mult=self.t_mult,
            eta_min=1e-6,
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
    # Checkpoint: save only trainable params + small essential buffers
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        saved: Dict[str, Any] = {}
        # Trainable parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    saved[key] = full_sd[key]
        # Essential small buffers; exclude large frozen embeddings
        large_frozen = {"string_embs", "esm2_embeddings"}
        for name, buf in self.named_buffers():
            leaf = name.split(".")[-1]
            if leaf not in large_frozen:
                key = prefix + name
                if key in full_sd:
                    saved[key] = full_sd[key]

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        self.print(
            f"Saving checkpoint: {n_trainable:,}/{n_total:,} trainable params "
            f"({100*n_trainable/n_total:.1f}%)"
        )
        return saved

    def load_state_dict(self, state_dict, strict=True):
        # strict=False: string_embs/esm2_embeddings not in checkpoint (recomputed in setup())
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-F1 averaged over all genes — matches calc_metric.py logic."""
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [N, 6640]
    n_genes = labels.shape[1]
    f1_vals: List[float] = []
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
    """Save test predictions in required TSV format (idx / input / prediction)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_ids = set()
    for i, (pid, sym) in enumerate(zip(pert_ids, symbols)):
        if pid not in seen_ids:
            seen_ids.add(pid)
            rows.append({
                "idx": pid,
                "input": sym,
                "prediction": json.dumps(preds[i].tolist()),
            })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}")


def _run_ensemble_inference(
    ckpt_paths: List[str],
    test_dataset: PerturbDataset,
    batch_size: int,
    device: torch.device,
    string_embs: torch.Tensor,
    esm2_embs: torch.Tensor,
    ensg2str_idx: Dict[str, int],
    ensg2esm_idx: Dict[str, int],
    hidden_dim: int = 384,
    n_blocks: int = 3,
    dropout: float = 0.30,
    head_dropout: float = 0.10,
) -> Tuple[List[str], List[str], np.ndarray]:
    """Run inference with each checkpoint and average logits (rank 0 only)."""
    all_preds_list: List[np.ndarray] = []
    pert_ids_out: List[str] = []
    symbols_out: List[str] = []

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)
    string_embs_dev = string_embs.to(device)
    esm2_embs_dev = esm2_embs.to(device)

    for ckpt_idx, ckpt_path in enumerate(ckpt_paths):
        print(f"Ensemble: checkpoint {ckpt_idx+1}/{len(ckpt_paths)}: {Path(ckpt_path).name}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        full_state_dict = ckpt.get("state_dict", ckpt)

        # Extract keys belonging to model (strip "model." prefix)
        model_state_dict = {}
        for k, v in full_state_dict.items():
            if k.startswith("model."):
                model_state_dict[k[len("model."):]] = v

        # Build fresh DualBranchModel and load weights
        nn_model = DualBranchModel(
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            dropout=dropout,
            head_dropout=head_dropout,
        )
        missing, unexpected = nn_model.load_state_dict(model_state_dict, strict=False)
        if missing:
            print(f"  Warning: missing keys: {missing[:3]}")
        nn_model.eval()
        nn_model.to(device)
        nn_model = nn_model.float()

        ckpt_preds: List[np.ndarray] = []

        with torch.no_grad():
            for batch in test_loader:
                pids = batch["pert_id"]
                str_idx = torch.tensor(
                    [ensg2str_idx.get(pid, -1) for pid in pids], dtype=torch.long
                ).to(device)
                esm_idx = torch.tensor(
                    [ensg2esm_idx.get(pid, -1) for pid in pids], dtype=torch.long
                ).to(device)
                logits, _ = nn_model(pids, str_idx, string_embs_dev, esm2_embs_dev, esm_idx)
                ckpt_preds.append(logits.cpu().float().numpy())
                if ckpt_idx == 0:
                    pert_ids_out.extend(batch["pert_id"])
                    symbols_out.extend(batch["symbol"])

        ckpt_preds_arr = np.concatenate(ckpt_preds, axis=0)  # [N, 3, 6640]
        all_preds_list.append(ckpt_preds_arr)

        del nn_model, ckpt, full_state_dict, model_state_dict, ckpt_preds
        torch.cuda.empty_cache()

    # Average logits across all checkpoint predictions
    ensemble_preds = np.mean(all_preds_list, axis=0)  # [N, 3, 6640]
    print(f"Ensemble complete: averaged {len(all_preds_list)} checkpoint predictions")
    return pert_ids_out, symbols_out, ensemble_preds


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node1-3-3-1-1: ESM2+STRING Dual-Branch + Gated Fusion + Mixup"
    )
    p.add_argument("--micro-batch-size",    type=int,   default=16)
    p.add_argument("--global-batch-size",   type=int,   default=128)
    p.add_argument("--max-epochs",          type=int,   default=600)
    p.add_argument("--muon-lr",             type=float, default=0.01)
    p.add_argument("--adamw-lr",            type=float, default=3e-4)
    p.add_argument("--weight-decay",        type=float, default=8e-4)
    p.add_argument("--hidden-dim",          type=int,   default=384)
    p.add_argument("--n-blocks",            type=int,   default=3)
    p.add_argument("--dropout",             type=float, default=0.30)
    p.add_argument("--head-dropout",        type=float, default=0.10)
    p.add_argument("--t0",                  type=int,   default=80)
    p.add_argument("--t-mult",              type=int,   default=2)
    p.add_argument("--mixup-prob",          type=float, default=0.65)
    p.add_argument("--mixup-alpha",         type=float, default=0.2)
    p.add_argument("--early-stop-patience", type=int,   default=120)
    p.add_argument("--save-top-k",          type=int,   default=5,
                   help="Number of top checkpoints to save for test-time ensemble")
    p.add_argument("--num-workers",         type=int,   default=4)
    p.add_argument("--val-check-interval",  type=float, default=1.0)
    p.add_argument("--debug_max_step",      type=int,   default=None)
    p.add_argument("--fast_dev_run",        action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    pl.seed_everything(0)
    np.random.seed(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- DataModule ---
    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup("fit")

    # --- LightningModule ---
    model = PerturbModule(
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay,
        t0=args.t0,
        t_mult=args.t_mult,
        mixup_prob=args.mixup_prob,
        mixup_alpha=args.mixup_alpha,
    )

    # --- Trainer configuration ---
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
        limit_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        limit_train = 1.0
        limit_val = 1.0
        limit_test = 1.0
        max_steps = -1

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val_f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
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
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

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
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    # --- Fit ---
    trainer.fit(model, datamodule=datamodule)

    # --- Test ---
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

        # Top-K Checkpoint Ensemble (rank 0 only, after DDP teardown)
        if trainer.is_global_zero and args.save_top_k > 1:
            ckpt_dir = Path(checkpoint_cb.dirpath)

            # Use rglob to robustly find checkpoints regardless of directory structure
            all_ckpts = list(ckpt_dir.rglob("best-*.ckpt"))

            def _extract_f1(p: Path) -> float:
                """Extract val_f1 score from checkpoint path."""
                try:
                    stem = p.stem  # e.g., "best-001-0.5100"
                    parts = stem.split("-")
                    return float(parts[-1])
                except (ValueError, IndexError):
                    return 0.0

            ckpt_files = sorted(
                [str(c) for c in all_ckpts],
                key=lambda p: _extract_f1(Path(p)),
                reverse=True,
            )[:args.save_top_k]

            if len(ckpt_files) >= 2:
                print(f"Running top-{len(ckpt_files)} checkpoint ensemble...")

                # Recompute frozen embeddings for ensemble inference
                from transformers import AutoModel
                gnn = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
                gnn.eval()
                graph = torch.load(
                    STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False
                )
                edge_index = graph["edge_index"]
                edge_weight = graph.get("edge_weight", None)
                with torch.no_grad():
                    gnn_out = gnn(edge_index=edge_index, edge_weight=edge_weight)
                string_embs = gnn_out.last_hidden_state.detach().float().cpu()
                del gnn, gnn_out

                # ESM2 embeddings are already stored in the model's buffer
                # We can retrieve them from the loaded model state
                esm2_embs = model.esm2_embeddings.cpu()
                ensg2str_idx = model.ensg2str_idx
                ensg2esm_idx = model.ensg2esm_idx

                # Ensure test dataloader is initialized (setup("fit") leaves test_ds as None)
                if datamodule.test_ds is None:
                    datamodule.setup("test")
                test_ds = datamodule.test_ds

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                pert_ids, symbols, ensemble_preds = _run_ensemble_inference(
                    ckpt_paths=ckpt_files,
                    test_dataset=test_ds,
                    batch_size=args.micro_batch_size * 2,
                    device=device,
                    string_embs=string_embs,
                    esm2_embs=esm2_embs,
                    ensg2str_idx=ensg2str_idx,
                    ensg2esm_idx=ensg2esm_idx,
                    hidden_dim=args.hidden_dim,
                    n_blocks=args.n_blocks,
                    dropout=args.dropout,
                    head_dropout=args.head_dropout,
                )

                out_path = Path(__file__).parent / "run" / "test_predictions.tsv"
                _save_test_predictions(pert_ids, symbols, ensemble_preds, out_path)
                print(f"Ensemble predictions saved to {out_path}")
            else:
                print(f"Only {len(ckpt_files)} checkpoints found, skipping ensemble.")


if __name__ == "__main__":
    main()
