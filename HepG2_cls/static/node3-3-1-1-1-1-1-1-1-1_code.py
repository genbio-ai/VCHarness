"""Node 1-2: ESM2-650M LoRA + STRING_GNN Gated Fusion
               + Focal Loss (γ=2.5) — NO Mixup + Extended Patience=50
================================================================
Parent  : node3-3-1-1-1-1-1-1-1  (STRING-only + Muon + FocalLoss(γ=2.5), test F1=0.4862)
Tree Best: node4-1-1-1-1  (ESM2-650M LoRA + STRING_GNN, FocalLoss(γ=2.0), Mixup α=0.2, test F1=0.5072)

Core Motivation:
The parent node (STRING-only, F1=0.4862) has reached the ceiling for STRING-only
feature representation (~0.486-0.488). The parent's feedback explicitly recommends
incorporating ESM2 protein language model features as the only path to meaningfully
exceed the tree best of 0.5072 (node4-1-1-1-1).

This node combines:
1. ESM2-650M LoRA (r=16) protein sequence embeddings — captures biochemical/functional
   properties of the perturbed gene from protein sequence
2. STRING_GNN frozen PPI graph embeddings — captures network topology context
3. Focal loss γ=2.5 (from parent, proven superior to node4's γ=2.0 in STRING lineage)
4. NO Mixup (from parent — Mixup interferes with focal loss hard-example mining)
5. NO label smoothing (from STRING lineage — eliminates training loss floor)
6. Extended patience=50 (from parent — critical enabler for multi-phase LR refinement)
7. RLROP scheduler (from STRING lineage — more stable than cosine for this task)

Strategy: Apply the proven STRING-only training recipe (γ=2.5, no Mixup, patience=50)
to the ESM2+STRING dual-branch architecture that achieved the tree best.

Changes vs parent (node3-3-1-1-1-1-1-1-1):
1. ADD ESM2-650M LoRA branch (r=16, target: query/key/value/dense)
   - Mean-pooled protein sequence embeddings [B, 1280]
   - Sequences from hg38_gencode_protein.fa, truncated to 512 tokens
   - 99.7% coverage of dataset pert_ids
2. MODIFY architecture: Gated fusion of ESM2 [1280] + STRING_GNN [256] → 512-dim
   - Linear projection: ESM2 1280→256, STRING_GNN 256→256
   - Gated fusion with learnable gate (sigmoid)
   - 3×PreNormResBlock(512, 1024, dropout=0.30)
3. MODIFY head: hidden=512 → 6640×3 (with per-gene bias)
4. MODIFY LR: Split learning rates — ESM2 LoRA at 5e-5, STRING/head at 1e-4
5. REDUCE head_dropout: 0.05 → 0.3 (node4's next iteration suggestion was 0.35)
6. KEEP focal_gamma=2.5, NO Mixup, patience=50, WD=0.01, RLROP

Memory estimate:
- ESM2-650M bf16: ~1.3 GB base
- LoRA r=16 activations (bs=8, seq_len=512): ~4.4 GB activation
- STRING_GNN frozen: ~0.1 GB
- Output head + optimizer: ~1.5 GB
- Total: ~8 GB per GPU — well within H100 80GB

Preserved from parent/STRING lineage:
- Focal loss γ=2.5 with class weights [0.0477, 0.9282, 0.0241]
- RLROP threshold=1e-5 (prevents over-aggressive halvings)
- RLROP patience=12 (grandparent value)
- Weight decay=0.01 (proven optimal)
- Gradient clip=2.0
- max_epochs=200 (ESM2 training is slower; empirically 120-150 epochs is sufficient)
- val_f1 metric name (no slash)
- auto_insert_metric_name=False
- MuonWithAuxAdam dual optimizer (Muon for block matrices, AdamW for LoRA+embeddings+head)
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import re
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRING_GNN_DIR = "/home/Models/STRING_GNN"
PROTEIN_FASTA = "/home/data/genome/hg38_gencode_protein.fa"
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"  # 650M parameter model
N_GENES = 6640        # number of response genes per perturbation
N_CLASSES = 3         # down (-1→0), neutral (0→1), up (1→2)
GNN_DIM = 256         # STRING_GNN output embedding dimension
ESM2_DIM = 1280       # ESM2-650M hidden dimension
PROJ_DIM = 256        # Projection dim for both branches before fusion
FUSED_DIM = 512       # Fused dimension after gating (PROJ_DIM * 2)
HIDDEN_DIM = 512      # MLP hidden dimension
INNER_DIM = 1024      # MLP inner (expansion) dimension
MAX_SEQ_LEN = 512     # Max ESM2 sequence length (truncation)


# ---------------------------------------------------------------------------
# Protein FASTA Loading
# ---------------------------------------------------------------------------
def load_protein_sequences(fasta_path: str) -> Dict[str, str]:
    """Load protein sequences from FASTA, keyed by ENSG ID (without version).

    If a gene has multiple isoforms, keeps the longest one.
    """
    ensg_to_seq: Dict[str, str] = {}
    current_ensg: Optional[str] = None
    current_seq: List[str] = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous record
                if current_ensg and current_seq:
                    seq = ''.join(current_seq)
                    # Keep longest isoform
                    if current_ensg not in ensg_to_seq or len(seq) > len(ensg_to_seq[current_ensg]):
                        ensg_to_seq[current_ensg] = seq
                current_seq = []
                current_ensg = None
                # Parse ENSG ID from header (format: >ENSP...|ENST...|ENSG...|...)
                for part in line.split('|'):
                    p = part.strip()
                    if p.startswith('ENSG'):
                        current_ensg = p.split('.')[0]  # strip version
                        break
            else:
                current_seq.append(line)

    # Save last record
    if current_ensg and current_seq:
        seq = ''.join(current_seq)
        if current_ensg not in ensg_to_seq or len(seq) > len(ensg_to_seq[current_ensg]):
            ensg_to_seq[current_ensg] = seq

    return ensg_to_seq


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(torch.utils.data.Dataset):
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
        micro_batch_size: int = 8,
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

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
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
    """Pre-LayerNorm residual block (proven stable in node1-3-2 lineage)."""

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
    """Gated fusion of two projected embeddings.

    Architecture:
        esm2_proj  = Linear(esm2_dim → proj_dim)
        gnn_proj   = Linear(gnn_dim → proj_dim)
        gate       = sigmoid(Linear(proj_dim * 2 → 1))
        fused      = cat(gate * esm2_proj, (1-gate) * gnn_proj)  → [B, proj_dim * 2]

    This allows the model to learn task-adaptive weighting between protein
    sequence features and PPI network topology features.
    """

    def __init__(
        self,
        esm2_dim: int = ESM2_DIM,
        gnn_dim: int = GNN_DIM,
        proj_dim: int = PROJ_DIM,
    ) -> None:
        super().__init__()
        self.esm2_proj = nn.Linear(esm2_dim, proj_dim)
        self.gnn_proj = nn.Linear(gnn_dim, proj_dim)
        # Gate is computed from both projections concatenated
        self.gate_net = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, 1),
        )
        self.output_norm = nn.LayerNorm(proj_dim * 2)

    def forward(
        self, esm2_emb: torch.Tensor, gnn_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            esm2_emb: [B, esm2_dim]
            gnn_emb:  [B, gnn_dim]
        Returns:
            fused: [B, proj_dim * 2]
        """
        e = self.esm2_proj(esm2_emb)   # [B, proj_dim]
        g = self.gnn_proj(gnn_emb)     # [B, proj_dim]
        gate = torch.sigmoid(self.gate_net(torch.cat([e, g], dim=-1)))  # [B, 1]
        fused = torch.cat([gate * e, (1 - gate) * g], dim=-1)           # [B, proj_dim * 2]
        return self.output_norm(fused)


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
        head_dropout: float = 0.30,      # Increased from parent's 0.05 for ESM2 richness
        esm2_lr: float = 5e-5,           # Conservative LR for LoRA fine-tuning
        backbone_lr: float = 1e-4,       # LR for STRING_GNN + fusion + head
        weight_decay: float = 0.01,      # Proven optimal from STRING lineage
        focal_gamma: float = 2.5,        # From parent — better than node4's 2.0
        lora_r: int = 16,                # LoRA rank
        lora_alpha: int = 32,            # LoRA alpha
        lora_dropout: float = 0.05,      # LoRA dropout
        rlrop_factor: float = 0.5,
        rlrop_patience: int = 12,        # Proven value from grandparent
        rlrop_threshold: float = 1e-5,   # Proven value — prevents over-aggressive halvings
        min_lr: float = 1e-7,
        grad_clip_norm: float = 2.0,
        max_seq_len: int = MAX_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Populated in setup()
        self.esm2: Optional[nn.Module] = None
        self.tokenizer = None
        self.fusion: Optional[GatedFusion] = None
        self.blocks: Optional[nn.ModuleList] = None
        self.output_head: Optional[nn.Sequential] = None
        self.gene_bias: Optional[nn.Parameter] = None

        # STRING_GNN gene-ID → embedding-row index
        self.gnn_id_to_idx: Dict[str, int] = {}
        # Protein sequences for ESM2
        self.ensg_to_seq: Dict[str, str] = {}

        # Metric accumulators
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        """Build model: ESM2-650M LoRA + frozen STRING_GNN embeddings."""
        from transformers import AutoTokenizer, EsmModel
        from peft import LoraConfig, get_peft_model, TaskType

        hp = self.hparams

        # ------ Step 1: Load protein sequences ------
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.print("Loading protein sequences from FASTA …")
        self.ensg_to_seq = load_protein_sequences(PROTEIN_FASTA)
        self.print(f"Loaded {len(self.ensg_to_seq)} protein sequences")

        # ------ Step 2: Load ESM2 tokenizer (rank-0 first) ------
        if local_rank == 0:
            AutoTokenizer.from_pretrained(ESM2_MODEL)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)

        # ------ Step 3: Load ESM2 model with LoRA ------
        self.print(f"Loading ESM2-650M with LoRA (r={hp.lora_r}) …")
        esm2_base = EsmModel.from_pretrained(
            ESM2_MODEL, dtype=torch.float32, add_pooling_layer=False
        )

        # Configure LoRA targeting attention projections
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            target_modules=["query", "key", "value", "dense"],
            bias="none",
        )
        self.esm2 = get_peft_model(esm2_base, lora_config)
        self.esm2.print_trainable_parameters()

        # Enable gradient checkpointing for memory efficiency
        self.esm2.gradient_checkpointing_enable()

        # ------ Step 4: Load STRING_GNN frozen embeddings ------
        self.print("Loading STRING_GNN and computing frozen node embeddings …")
        gnn_model = torch.hub.load.__module__  # just to avoid bare import
        from transformers import AutoModel as _AutoModel
        gnn = _AutoModel.from_pretrained(
            STRING_GNN_DIR, trust_remote_code=True
        )
        gnn.eval()
        gnn = gnn.to(self.device)

        graph = torch.load(
            Path(STRING_GNN_DIR) / "graph_data.pt",
            map_location=self.device,
        )
        edge_index = graph["edge_index"].to(self.device)
        edge_weight = graph.get("edge_weight")
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        with torch.no_grad():
            gnn_out = gnn(edge_index=edge_index, edge_weight=edge_weight)

        # Register as non-trainable float32 buffer [18870, 256]
        all_emb = gnn_out.last_hidden_state.detach().float()
        self.register_buffer("gnn_embeddings", all_emb)

        # Free GNN model memory
        del gnn, gnn_out, graph, edge_index, edge_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.print(f"STRING_GNN embeddings shape: {all_emb.shape}")

        # Build ENSG-ID → row-index mapping
        node_names: List[str] = json.loads(
            (Path(STRING_GNN_DIR) / "node_names.json").read_text()
        )
        self.gnn_id_to_idx = {name: i for i, name in enumerate(node_names)}
        self.print(f"STRING_GNN covers {len(self.gnn_id_to_idx)} Ensembl gene IDs")

        # ------ Step 5: Build fusion + MLP head ------
        self.fusion = GatedFusion(
            esm2_dim=ESM2_DIM,
            gnn_dim=GNN_DIM,
            proj_dim=PROJ_DIM,
        )
        self.blocks = nn.ModuleList(
            [
                PreNormResBlock(hp.hidden_dim, hp.inner_dim, hp.dropout)
                for _ in range(hp.n_blocks)
            ]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hp.hidden_dim),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.hidden_dim, N_GENES * N_CLASSES),
        )

        # Per-gene additive bias
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES, N_CLASSES))

        # ------ Step 6: Class weights ------
        # class 0 = down-regulated  (4.77%)  → high weight
        # class 1 = neutral         (92.82%) → low weight
        # class 2 = up-regulated    (2.41%)  → highest weight
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
            f"Architecture: ESM2-650M LoRA(r={hp.lora_r}) [{ESM2_DIM}-dim] + "
            f"STRING_GNN [{GNN_DIM}-dim] → GatedFusion → "
            f"{hp.n_blocks}×PreNormResBlock({hp.hidden_dim},{hp.inner_dim}) "
            f"→ HeadDropout({hp.head_dropout}) → Linear({hp.hidden_dim},{N_GENES}×{N_CLASSES}) + gene_bias"
        )
        self.print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        self.print(f"FOCAL LOSS: gamma={hp.focal_gamma} (from parent, NO Mixup)")
        self.print(f"Split LR: ESM2 LoRA={hp.esm2_lr}, STRING+head={hp.backbone_lr}")

    # ------------------------------------------------------------------
    def _get_esm2_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Run ESM2 on protein sequences to get per-sample embeddings.

        Returns [B, ESM2_DIM] mean-pooled embeddings.
        Missing sequences receive zero vectors.
        """
        hp = self.hparams
        sequences = []
        for pid in pert_ids:
            seq = self.ensg_to_seq.get(pid, "")
            if seq:
                # Truncate to max_seq_len - 2 (for <cls> and <eos>)
                sequences.append(seq[:hp.max_seq_len - 2])
            else:
                sequences.append("M")  # Minimal fallback for missing sequences

        # Tokenize all sequences
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=hp.max_seq_len,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward through ESM2
        outputs = self.esm2(**inputs, output_hidden_states=False)
        hidden = outputs.last_hidden_state  # [B, seq_len, ESM2_DIM]

        # Mean pool over valid tokens (exclude <cls>, <eos>, <pad>)
        special_ids = torch.tensor(
            [self.tokenizer.cls_token_id, self.tokenizer.eos_token_id,
             self.tokenizer.pad_token_id],
            device=self.device
        )
        mask = ~torch.isin(inputs["input_ids"], special_ids)  # [B, seq_len]
        mask_f = mask.float().unsqueeze(-1)  # [B, seq_len, 1]
        sum_hidden = (hidden * mask_f).sum(dim=1)  # [B, ESM2_DIM]
        count = mask_f.sum(dim=1).clamp(min=1.0)   # [B, 1]
        emb = (sum_hidden / count).float()          # [B, ESM2_DIM]

        # Zero out embeddings for missing sequences
        for i, pid in enumerate(pert_ids):
            if pid not in self.ensg_to_seq:
                emb[i].zero_()

        return emb

    def _get_gnn_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Lookup frozen STRING_GNN embeddings for ENSG IDs."""
        emb_list: List[torch.Tensor] = []
        for pid in pert_ids:
            row = self.gnn_id_to_idx.get(pid)
            if row is not None:
                emb_list.append(self.gnn_embeddings[row])
            else:
                emb_list.append(
                    torch.zeros(GNN_DIM, device=self.device, dtype=torch.float32)
                )
        return torch.stack(emb_list, dim=0)  # [B, 256]

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits of shape [B, N_CLASSES, N_GENES]."""
        esm2_emb = self._get_esm2_emb(pert_ids)   # [B, 1280]
        gnn_emb = self._get_gnn_emb(pert_ids)      # [B, 256]

        x = self.fusion(esm2_emb, gnn_emb)         # [B, 512]

        for block in self.blocks:
            x = block(x)                            # [B, 512]

        logits = self.output_head(x)                # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)  # [B, 3, 6640]
        logits = logits + self.gene_bias.T.unsqueeze(0)  # + [1, 3, 6640]
        return logits

    def _compute_focal_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss with class weights on [B, N_CLASSES, N_GENES] logits."""
        hp = self.hparams
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)

        if hp.focal_gamma == 0.0:
            return F.cross_entropy(logits_flat, labels_flat, weight=self.class_weights)

        # Focal loss: FL(p_t) = -(1 - p_t)^gamma * w_c * log(p_t)
        ce_per_sample = F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            reduction="none",
        )

        with torch.no_grad():
            probs = F.softmax(logits_flat, dim=-1)
            pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt.clamp(min=1e-8)) ** hp.focal_gamma
        return (focal_weight * ce_per_sample).mean()

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pert_ids = batch["pert_id"]
        labels = batch["label"]

        logits = self(pert_ids)
        loss = self._compute_focal_loss(logits, labels)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        loss = self._compute_focal_loss(logits, batch["label"])
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
            self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        else:
            preds_np = preds_local.numpy()
            labels_np = labels_local.numpy()
            f1 = _compute_per_gene_f1(preds_np, labels_np)
            self.log("val_f1", f1, prog_bar=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        self._test_preds.append(logits.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        self._test_preds.clear()

        # Gather across all ranks
        gathered = self.all_gather(preds_local)  # [world_size, N_local, 3, 6640]
        all_preds = gathered.view(-1, N_CLASSES, N_GENES)  # [N_total, 3, 6640]

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

            # De-duplicate
            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            preds_np = all_preds.cpu().numpy()
            for i, pid in enumerate(all_pert_ids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_ids.append(pid)
                    dedup_syms.append(all_symbols[i])
                    dedup_preds.append(preds_np[i])

            self._current_test_preds_for_ensemble = np.stack(dedup_preds, axis=0)
            self._current_test_ids = dedup_ids
            self._current_test_syms = dedup_syms

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        from muon import MuonWithAuxAdam

        hp = self.hparams

        # Parameter groups:
        # Group 1 (ESM2 LoRA): All LoRA parameters → AdamW with esm2_lr
        # Group 2 (MLP blocks, 2D matrices): Muon with backbone_lr
        # Group 3 (Everything else): AdamW with backbone_lr
        #   (STRING frozen, fusion, norms, biases, output head, gene_bias)

        # Identify LoRA parameters
        esm2_lora_params = [
            p for name, p in self.named_parameters()
            if p.requires_grad and ("lora_A" in name or "lora_B" in name)
        ]
        esm2_lora_param_ids = set(id(p) for p in esm2_lora_params)

        # Muon for 2D weight matrices in the MLP residual blocks
        muon_params = [
            p for name, p in self.blocks.named_parameters()
            if p.ndim >= 2 and p.requires_grad
        ]
        muon_param_ids = set(id(p) for p in muon_params)

        # AdamW for everything else (norms, biases, gene_bias, fusion, output_head)
        adamw_backbone_params = [
            p for name, p in self.named_parameters()
            if p.requires_grad
            and id(p) not in esm2_lora_param_ids
            and id(p) not in muon_param_ids
        ]

        param_groups = [
            # ESM2 LoRA parameters — AdamW with low LR
            dict(
                params=esm2_lora_params,
                use_muon=False,
                lr=hp.esm2_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.weight_decay,
            ),
            # Hidden block weight matrices — Muon with backbone LR
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.backbone_lr,
                weight_decay=hp.weight_decay,
                momentum=0.95,
            ),
            # All other trainable parameters — AdamW with backbone LR
            dict(
                params=adamw_backbone_params,
                use_muon=False,
                lr=hp.backbone_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.weight_decay,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=hp.rlrop_factor,
            patience=hp.rlrop_patience,
            min_lr=hp.min_lr,
            threshold=hp.rlrop_threshold,
            threshold_mode="abs",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters (LoRA + MLP + head) to save disk space."""
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
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%), plus {total_buffers} buffer values"
        )

        return trainable_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """Load partial checkpoint (trainable params only)."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node1-2: ESM2-650M LoRA(r=16) + STRING_GNN Gated Fusion + "
                    "FocalLoss(gamma=2.5) + NO Mixup + Patience=50"
    )
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--esm2-lr", type=float, default=5e-5)
    p.add_argument("--backbone-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--focal-gamma", type=float, default=2.5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--head-dropout", type=float, default=0.30)
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--inner-dim", type=int, default=INNER_DIM)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--rlrop-factor", type=float, default=0.5)
    p.add_argument("--rlrop-patience", type=int, default=12)
    p.add_argument("--rlrop-threshold", type=float, default=1e-5)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--grad-clip-norm", type=float, default=2.0)
    p.add_argument("--early-stop-patience", type=int, default=50)
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    p.add_argument("--val_check_interval", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = (Path(__file__).parent / ".." / ".." / "data").resolve()
    output_dir = (Path(__file__).parent / "run").resolve()
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
        esm2_lr=args.esm2_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        rlrop_factor=args.rlrop_factor,
        rlrop_patience=args.rlrop_patience,
        rlrop_threshold=args.rlrop_threshold,
        min_lr=args.min_lr,
        grad_clip_norm=args.grad_clip_norm,
        max_seq_len=args.max_seq_len,
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
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
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
        strategy=DDPStrategy(
            find_unused_parameters=False,  # All parameters participate in forward pass
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
        deterministic=False,  # ESM2 with gradient checkpointing may not be deterministic
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.fit(model, datamodule=datamodule)

    # ------------------------------------------------------------------
    # Test: Single best checkpoint
    # ------------------------------------------------------------------
    if args.fast_dev_run or args.debug_max_step is not None:
        print("\n=== DEBUG MODE: Single checkpoint test ===")
        trainer.test(model, datamodule=datamodule)
        if hasattr(model, '_current_test_preds_for_ensemble') and trainer.is_global_zero:
            _save_test_predictions(
                pert_ids=model._current_test_ids,
                symbols=model._current_test_syms,
                preds=model._current_test_preds_for_ensemble,
                out_path=output_dir / "test_predictions.tsv",
            )
        test_results = [{"note": "debug_mode_single_ckpt"}]
    else:
        print("\n=== PRODUCTION MODE: Single best checkpoint test ===")
        checkpoint_dir = output_dir / "checkpoints"
        ckpt_files = sorted(checkpoint_dir.glob("best-*.ckpt"))

        if ckpt_files:
            def extract_f1(path):
                m = re.search(r'best-[^-]+-([0-9]+\.[0-9]{4})\.ckpt', str(path))
                if m:
                    return float(m.group(1))
                m2 = re.search(r'val_f1[=_]([0-9]+\.[0-9]+)', str(path))
                if m2:
                    return float(m2.group(1))
                return 0.0

            scored = [(f, extract_f1(f)) for f in ckpt_files]
            scored.sort(key=lambda x: x[1], reverse=True)
            best_ckpt = scored[0][0]
            print(f"Best checkpoint: {best_ckpt.name} (val_f1={scored[0][1]:.4f})")

            trainer.test(model, datamodule=datamodule, ckpt_path=str(best_ckpt))
        else:
            print("WARNING: No checkpoint files found. Using best checkpoint via Lightning.")
            trainer.test(model, datamodule=datamodule, ckpt_path='best')

        if hasattr(model, '_current_test_preds_for_ensemble') and trainer.is_global_zero:
            _save_test_predictions(
                pert_ids=model._current_test_ids,
                symbols=model._current_test_syms,
                preds=model._current_test_preds_for_ensemble,
                out_path=output_dir / "test_predictions.tsv",
            )
        test_results = [{"mode": "single_best_checkpoint"}]

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results saved → {score_path}")


if __name__ == "__main__":
    main()
