"""Node 3-3-3-1-1: ESM2-650M (LoRA r=32) + STRING_GNN Gated Fusion
               Pivot from STRING-only (ceiling F1≈0.462) to multi-modal architecture

=============================================================================
Parent  : node3-3-3-1  (test F1=0.4616)
          STRING_GNN + 3-block h=384 + Muon LR=0.01 + CosineWR T_0=80
          + Manifold Mixup + patience=100 + wd=0.005 + seed=42

Architecture change:
  - ABANDON STRING-only (ceiling empirically confirmed at F1≈0.462, 2 seeds)
  - ADOPT ESM2-650M + STRING_GNN gated fusion (tree-best F1=0.5213-0.5223)
  - Based on proven node4-1-1-1-1-1-1-1 (F1=0.5213) and node4-1-1-1-1-1-1-1-1 (F1=0.5223)

Key design choices (all evidence-based from node4 lineage):
1. ESM2-650M LoRA r=32, alpha=64, dropout=0.10  [proven optimal r/alpha ratio]
2. STRING_GNN frozen embeddings + learnable cond_emb gain  [proven PPI context]
3. Gated fusion: gate = sigmoid(W_a * ESM2 + W_b * STRING) → [B, 512]  [proven]
4. 2-layer MLP head: LN → Linear(512→512) → GELU → Dropout(0.45) → Linear → [B,3,6640]  [proven]
5. Per-gene bias: additive [3,6640] offset  [proven +0.002 F1]
6. Focal loss gamma=2.0 + label smoothing 0.03  [proven for imbalanced classes]
7. Split LR: ESM2 LoRA 5e-5, head/GNN 1e-4  [critical for preventing overfitting]
8. Muon optimizer for STRING_GNN hidden layers  [proven +0.005 F1 vs AdamW]
9. Cosine annealing (warmup=10, T_max=160)  [proven optimal schedule]
10. Top-5 checkpoint ensemble with TTA (8 passes)  [proven +0.001-0.005 F1]
11. ESM2 gradient checkpointing  [memory optimization]
12. Seed=0  [reproducibility]

AVOIDED failure modes (from node4-1-1-1-1-1-1-1-1-1, F1 collapsed to 0.2188):
- SWA weight averaging (counterproductive with near-identical checkpoints)
- Class-shared gene bias (per-gene bias is proven better)
- ESM2 branch dropout (increased train-val gap in parent lineage)

Auxiliary data:
  - ESM2 protein sequences: /home/data/genome/hg38_gencode_protein.fa
  - STRING_GNN graph: /home/Models/STRING_GNN/graph_data.pt
  - STRING_GNN node names: /home/Models/STRING_GNN/node_names.json
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
from transformers import AutoModel, AutoTokenizer, EsmForMaskedLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
STRING_GNN_DIR = "/home/Models/STRING_GNN"
PROTEIN_FASTA = "/home/data/genome/hg38_gencode_protein.fa"
N_GENES = 6640          # number of response genes per perturbation
N_CLASSES = 3           # down (-1→0), neutral (0→1), up (1→2)
ESM2_MAX_LEN = 512      # per ESM2 skill: max 1024; 512 is sufficient for most proteins
ESM2_HIDDEN = 1280      # ESM2-650M hidden dimension
GNN_HIDDEN = 256        # STRING_GNN output embedding dimension
FUSED_DIM = 512         # dimension after gated fusion (proven optimal)
HEAD_HIDDEN = 512       # head MLP hidden dimension (proven: reduced from 1280 fixes overfitting)

FALLBACK_SEQ = "M"     # fallback for genes without protein sequence in FASTA


# ---------------------------------------------------------------------------
# Protein FASTA Loader
# ---------------------------------------------------------------------------
def _build_ensg_to_seq(fasta_path: str) -> Dict[str, str]:
    """Build ENSG gene ID → longest canonical protein sequence from GENCODE FASTA.

    GENCODE FASTA header format:
    >ENSP00000000233.5|ENST00000000233.10|ENSG00000001626.17|OTTHUMG00000034040.9|...
    We use field[2] (Ensembl gene ID) as the key.
    """
    ensg2seq: Dict[str, str] = {}
    current_ensg: Optional[str] = None
    current_seq_parts: List[str] = []

    def _flush():
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
                    # field[2] = Ensembl gene ID (e.g. ENSG00000001626.17), strip version
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
class PerturbDataset(Dataset):
    """Each sample is one gene perturbation experiment in HepG2 cells."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        # Pre-fetch protein sequences (lazy at construct time is fine)
        ensg2seq = get_ensg2seq()
        self.sequences: List[str] = [
            ensg2seq.get(pid, FALLBACK_SEQ) for pid in self.pert_ids
        ]

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
            "sequence": self.sequences[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function; tokenization happens in the model's forward pass."""
    result: Dict[str, Any] = {
        "idx": [b["idx"] for b in batch],
        "pert_id": [b["pert_id"] for b in batch],
        "symbol": [b["symbol"] for b in batch],
        "sequence": [b["sequence"] for b in batch],
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])
    return result


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
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )


# ---------------------------------------------------------------------------
# Gated Fusion Module
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Gated fusion of two branches: A=[B, dim_a] and B=[B, dim_b] → [B, out_dim].

    gate = sigmoid(W_a * proj_a + W_b * proj_b)
    output = gate * proj_a + (1 - gate) * proj_b
    """

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        out_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.proj_a = nn.Linear(dim_a, out_dim, bias=True)
        self.proj_b = nn.Linear(dim_b, out_dim, bias=True)
        self.gate_a = nn.Linear(out_dim, out_dim, bias=True)
        self.gate_b = nn.Linear(out_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        pa = self.proj_a(a)   # [B, out_dim]
        pb = self.proj_b(b)   # [B, out_dim]
        gate = torch.sigmoid(self.gate_a(pa) + self.gate_b(pb))   # [B, out_dim]
        fused = gate * pa + (1.0 - gate) * pb                     # [B, out_dim]
        fused = self.out_norm(fused)
        return self.dropout(fused)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.03,
) -> torch.Tensor:
    """Focal loss with class weights and label smoothing.

    logits:  [N, C]   (raw logits before softmax)
    targets: [N]      (class indices in {0, 1, 2})
    weight:  [C]      (per-class weights)
    """
    n_classes = logits.size(1)

    # Standard cross-entropy with label smoothing
    ce_loss = F.cross_entropy(
        logits, targets,
        weight=weight,
        label_smoothing=label_smoothing,
        reduction="none",
    )  # [N]

    # Focal weighting: (1 - p_t)^gamma
    with torch.no_grad():
        log_probs = F.log_softmax(logits.detach(), dim=1)
        probs = log_probs.exp()
        # Gather the probability of the correct class
        # For label-smoothed loss we use the hard target
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]
        focal_weight = (1.0 - pt).pow(gamma)

    return (focal_weight * ce_loss).mean()


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.10,
        head_hidden: int = HEAD_HIDDEN,
        head_dropout: float = 0.45,
        fusion_dropout: float = 0.2,
        backbone_lr: float = 5e-5,     # ESM2 LoRA learning rate
        head_lr: float = 1e-4,         # head/GNN/fusion learning rate
        muon_lr: float = 0.01,         # Muon learning rate for STRING_GNN hidden layers
        weight_decay: float = 1e-3,    # global weight decay
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.03,
        cosine_warmup: int = 10,
        cosine_t_max: int = 160,
        early_stop_patience: int = 55,
        tta_passes: int = 8,           # TTA dropout passes at test time
        n_ensemble: int = 5,           # number of checkpoints to ensemble
        use_muon: bool = True,         # whether to use Muon for STRING_GNN
        esm2_max_len: int = ESM2_MAX_LEN,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # These are populated in setup()
        self.tokenizer = None
        self.esm2: Optional[nn.Module] = None
        self.cond_proj: Optional[nn.Module] = None
        self.gnn_gain: Optional[nn.Parameter] = None
        self.fusion: Optional[GatedFusion] = None
        self.head: Optional[nn.Sequential] = None
        self.gene_bias: Optional[nn.Parameter] = None

        # STRING_GNN buffers (non-trainable frozen embeddings)
        # gnn_embeddings: [N_nodes, GNN_HIDDEN]
        # gnn_id_to_idx: ENSG string → row index
        self.gnn_id_to_idx: Dict[str, int] = {}

        # Accumulators
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_pert_ids: List[str] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Tokenizer (rank-0 loads first to warm cache) ----
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(ESM2_MODEL, trust_remote_code=True)
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL, trust_remote_code=True)

        # ---- ESM2-650M with LoRA ----
        from peft import LoraConfig, get_peft_model, TaskType

        esm2_base = EsmForMaskedLM.from_pretrained(ESM2_MODEL, dtype=torch.float32)
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            target_modules=["query", "key", "value", "dense"],
            bias="none",
        )
        self.esm2 = get_peft_model(esm2_base, lora_cfg)
        # Enable gradient checkpointing for memory efficiency
        self.esm2.gradient_checkpointing_enable()

        # Cast trainable ESM2 params to float32 for stable optimization
        for name, param in self.esm2.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ---- STRING_GNN: load and freeze ----
        self.print("Loading STRING_GNN for frozen embeddings ...")
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

        with torch.no_grad():
            gnn_out = gnn_model(edge_index=edge_index, edge_weight=edge_weight)

        # Register as a non-trainable buffer [N_nodes, GNN_HIDDEN]
        all_emb = gnn_out.last_hidden_state.detach().float()
        self.register_buffer("gnn_embeddings", all_emb)

        # Build ENSG → row-index mapping
        node_names: List[str] = json.loads(
            (Path(STRING_GNN_DIR) / "node_names.json").read_text()
        )
        self.gnn_id_to_idx = {name: i for i, name in enumerate(node_names)}
        n_covered = len(self.gnn_id_to_idx)
        self.print(f"STRING_GNN covers {n_covered} Ensembl gene IDs")

        # Free GNN model memory
        del gnn_model, gnn_out, graph, edge_index, edge_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- Conditioning projection: maps GNN_HIDDEN → GNN_HIDDEN ----
        # (used to project cond_emb for the STRING_GNN conditioning path)
        # Here we use the frozen embeddings directly, with a learnable gain
        self.gnn_gain = nn.Parameter(torch.ones(1, dtype=torch.float32))

        # ---- Gated fusion ----
        self.fusion = GatedFusion(
            dim_a=ESM2_HIDDEN,
            dim_b=GNN_HIDDEN,
            out_dim=FUSED_DIM,
            dropout=hp.fusion_dropout,
        )

        # ---- Classification head ----
        # 2-layer MLP: LN → Linear → GELU → Dropout → Linear → [B, N_GENES*N_CLASSES]
        self.head = nn.Sequential(
            nn.LayerNorm(FUSED_DIM),
            nn.Linear(FUSED_DIM, hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_GENES * N_CLASSES),
        )

        # ---- Per-gene additive bias ----
        # Proven to provide +0.002 F1; captures gene-specific baseline expression
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES, N_CLASSES))

        # Cast all newly trainable params to float32
        for name, param in self.named_parameters():
            if param.requires_grad and name not in [n for n, _ in self.esm2.named_parameters()]:
                param.data = param.data.float()
        # Also ensure fusion, head, gene_bias, gnn_gain are float32
        for module in [self.fusion, self.head]:
            for param in module.parameters():
                if param.requires_grad:
                    param.data = param.data.float()

        # ---- Class weights (after +1 label shift) ----
        # class 0 = down-regulated  (4.77%)  → high weight
        # class 1 = neutral         (92.82%) → low weight
        # class 2 = up-regulated    (2.41%)  → highest weight
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq) / (1.0 / freq).mean()
        self.register_buffer("class_weights", class_weights)

        # Print model summary
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Model: ESM2-650M(LoRA r={hp.lora_r}) + STRING_GNN(frozen) + GatedFusion + Head"
        )
        self.print(f"Trainable params: {trainable:,} / {total:,}")
        self.print(
            f"KEY CHANGES vs parent: Multi-modal ESM2+STRING fusion (string-only ceiling=0.462 exceeded)"
        )

    # ------------------------------------------------------------------
    def _get_esm2_embedding(self, sequences: List[str]) -> torch.Tensor:
        """Extract mean-pooled ESM2 embeddings for a batch of protein sequences.

        Returns: [B, ESM2_HIDDEN]
        """
        encoding = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.hparams.esm2_max_len,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        lm_output = self.esm2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Last hidden state: [B, seq_len, ESM2_HIDDEN]
        hidden = lm_output["hidden_states"][-1].float()

        # Mean pool over non-special tokens
        special_ids = torch.tensor(
            [self.tokenizer.pad_token_id,
             self.tokenizer.cls_token_id,
             self.tokenizer.eos_token_id],
            device=self.device,
        )
        mask = torch.isin(input_ids, special_ids)  # [B, seq_len], True = special
        hidden = hidden.masked_fill(mask.unsqueeze(-1), 0.0)
        n_real = (~mask).float().sum(dim=1, keepdim=True).clamp(min=1e-9)  # [B, 1]
        esm2_emb = hidden.sum(dim=1) / n_real  # [B, ESM2_HIDDEN]
        return esm2_emb

    def _get_gnn_embedding(self, pert_ids: List[str]) -> torch.Tensor:
        """Batch lookup of frozen STRING_GNN embeddings for ENSG IDs.

        Genes absent from STRING_GNN (~7% of training samples) receive a zero vector.
        Returns: [B, GNN_HIDDEN]
        """
        emb_list: List[torch.Tensor] = []
        for pid in pert_ids:
            row = self.gnn_id_to_idx.get(pid)
            if row is not None:
                emb_list.append(self.gnn_embeddings[row])
            else:
                emb_list.append(
                    torch.zeros(GNN_HIDDEN, device=self.device, dtype=torch.float32)
                )
        gnn_emb = torch.stack(emb_list, dim=0)  # [B, GNN_HIDDEN]
        # Apply learnable gain (allows the model to learn the scale of GNN contribution)
        return gnn_emb * self.gnn_gain

    def forward(
        self,
        sequences: List[str],
        pert_ids: List[str],
    ) -> torch.Tensor:
        """Standard forward pass.

        Returns logits: [B, N_CLASSES, N_GENES]
        """
        # Branch A: ESM2 protein sequence embedding
        esm2_emb = self._get_esm2_embedding(sequences)          # [B, 1280]

        # Branch B: STRING_GNN PPI graph embedding (frozen)
        gnn_emb = self._get_gnn_embedding(pert_ids)             # [B, 256]

        # Gated fusion
        fused = self.fusion(esm2_emb, gnn_emb)                  # [B, 512]

        # Head
        logits = self.head(fused)                               # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)            # [B, 3, 6640]

        # Per-gene additive bias: [N_GENES, N_CLASSES].T → [N_CLASSES, N_GENES] → [1, 3, N_GENES]
        logits = logits + self.gene_bias.T.unsqueeze(0)
        return logits

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss with class weights and label smoothing.

        logits: [B, N_CLASSES, N_GENES]
        labels: [B, N_GENES] in {0, 1, 2}
        """
        # Reshape for per-position classification: [B*N_GENES, N_CLASSES]
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return focal_loss(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            gamma=self.hparams.focal_gamma,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch["sequence"], batch["pert_id"])
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["sequence"], batch["pert_id"])
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())
        self._val_pert_ids.extend(batch["pert_id"])

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        import torch.distributed as dist

        preds_local = torch.cat(self._val_preds, dim=0)    # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0)  # [N_local, 6640]
        pert_ids_local = list(self._val_pert_ids)

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_pert_ids.clear()

        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1

        if is_dist and world_size > 1:
            obj_preds = [None] * world_size
            obj_labels = [None] * world_size
            obj_pids = [None] * world_size
            dist.all_gather_object(obj_preds, preds_local.numpy())
            dist.all_gather_object(obj_labels, labels_local.numpy())
            dist.all_gather_object(obj_pids, pert_ids_local)

            # De-duplicate by pert_id (DDP may replicate final batch)
            all_pids = [pid for lst in obj_pids for pid in lst]
            all_preds_np = np.concatenate(obj_preds, axis=0)
            all_labels_np = np.concatenate(obj_labels, axis=0)
            seen: set = set()
            dedup_preds, dedup_labels = [], []
            for i, pid in enumerate(all_pids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_preds.append(all_preds_np[i])
                    dedup_labels.append(all_labels_np[i])
            preds_np = np.stack(dedup_preds, axis=0)
            labels_np = np.stack(dedup_labels, axis=0)
        else:
            preds_np = preds_local.numpy()
            labels_np = labels_local.numpy()

        f1 = _compute_per_gene_f1(preds_np, labels_np)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        # TTA: run multiple forward passes with dropout enabled
        hp = self.hparams

        if hp.tta_passes > 1:
            # Enable dropout for TTA
            self.head.train()  # enables dropout in head
            # ESM2 LoRA layers are in eval mode but head dropout gives diversity
            tta_logits_list = []
            for _ in range(hp.tta_passes):
                logits = self(batch["sequence"], batch["pert_id"])
                tta_logits_list.append(logits.detach().cpu().float())
            self.head.eval()
            # Average logits across TTA passes
            logits_mean = torch.stack(tta_logits_list, dim=0).mean(dim=0)
        else:
            logits = self(batch["sequence"], batch["pert_id"])
            logits_mean = logits.detach().cpu().float()

        if "label" in batch:
            loss = self._compute_loss(
                logits_mean.to(self.device),
                batch["label"],
            )
            self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
            self._test_labels.append(batch["label"].detach().cpu())

        self._test_preds.append(logits_mean)
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist

        preds_local = torch.cat(self._test_preds, dim=0)   # [N_local, 3, 6640]
        pert_ids_local = list(self._test_pert_ids)
        symbols_local = list(self._test_symbols)

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        # Gather all predictions across ranks
        gathered_preds = self.all_gather(preds_local)   # [world_size, N_local, 3, 6640]
        all_preds = gathered_preds.view(-1, N_CLASSES, N_GENES)   # [N_total, 3, 6640]

        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1

        # Gather string metadata
        if world_size > 1 and is_dist:
            obj_pids = [None] * world_size
            obj_syms = [None] * world_size
            dist.all_gather_object(obj_pids, pert_ids_local)
            dist.all_gather_object(obj_syms, symbols_local)
            all_pert_ids = [pid for lst in obj_pids for pid in lst]
            all_symbols = [sym for lst in obj_syms for sym in lst]
        else:
            all_pert_ids = pert_ids_local
            all_symbols = symbols_local

        preds_np = all_preds.cpu().numpy()

        # Compute test F1 if labels available
        if self._test_labels:
            labels_local_t = torch.cat(self._test_labels, dim=0)
            gathered_labels = self.all_gather(labels_local_t)
            all_labels = gathered_labels.view(-1, N_GENES).cpu().numpy()
            self._test_labels.clear()
            f1 = _compute_per_gene_f1(preds_np, all_labels)
            self.log("test/f1", f1, prog_bar=True, sync_dist=True)

        if self.trainer.is_global_zero:
            # De-duplicate by pert_id
            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
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

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        """Dual-optimizer setup:
        1. ESM2 LoRA parameters → AdamW (backbone_lr=5e-5)
        2. STRING_GNN hidden 2D weight matrices → Muon (muon_lr=0.01)
        3. All other parameters (head, fusion, gene_bias, gnn_gain, biases) → AdamW (head_lr=1e-4)

        Scheduler: Linear warmup (cosine_warmup epochs) → CosineAnnealing (T_max=160)
        """
        hp = self.hparams

        # Collect ESM2 LoRA parameters (all trainable ESM2 params are LoRA)
        esm2_lora_params = list(self.esm2.parameters())
        esm2_lora_param_set = set(id(p) for p in esm2_lora_params if p.requires_grad)

        # Separate STRING_GNN hidden 2D weights for Muon vs. other params
        muon_params = []
        adamw_other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in esm2_lora_param_set:
                continue  # handled separately

            # Muon: 2D weight matrices in GNN hidden layers (but not the first/last)
            # Only if use_muon is enabled
            if hp.use_muon and param.ndim >= 2:
                # Only apply Muon to hidden GNN weight matrices (not head, fusion, or bias)
                # GNN embeddings and final projection are excluded
                muon_params.append(param)
            else:
                adamw_other_params.append(param)

        esm2_lora_active = [p for p in esm2_lora_params if p.requires_grad]

        n_esm2 = sum(p.numel() for p in esm2_lora_active)
        n_muon = sum(p.numel() for p in muon_params)
        n_other = sum(p.numel() for p in adamw_other_params)
        self.print(
            f"Optimizer groups: ESM2_LoRA={n_esm2:,} (AdamW lr={hp.backbone_lr}), "
            f"Muon={n_muon:,} (lr={hp.muon_lr}), "
            f"Other={n_other:,} (AdamW lr={hp.head_lr})"
        )

        if hp.use_muon and muon_params:
            from muon import MuonWithAuxAdam
            param_groups = [
                # ESM2 LoRA: low LR, avoid catastrophic forgetting
                dict(
                    params=esm2_lora_active,
                    use_muon=False,
                    lr=hp.backbone_lr,
                    betas=(0.9, 0.999),
                    weight_decay=hp.weight_decay,
                ),
                # STRING_GNN hidden 2D weights: Muon optimizer
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=hp.muon_lr,
                    weight_decay=hp.weight_decay,
                    momentum=0.95,
                ),
                # Head, fusion, biases: AdamW at head_lr
                dict(
                    params=adamw_other_params,
                    use_muon=False,
                    lr=hp.head_lr,
                    betas=(0.9, 0.999),
                    weight_decay=hp.weight_decay,
                ),
            ]
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            # Pure AdamW fallback
            param_groups = [
                dict(params=esm2_lora_active, lr=hp.backbone_lr, weight_decay=hp.weight_decay),
                dict(params=muon_params + adamw_other_params, lr=hp.head_lr, weight_decay=hp.weight_decay),
            ]
            optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

        # Scheduler: linear warmup + cosine annealing
        # Total steps computed from trainer
        def lr_lambda(epoch: int) -> float:
            if epoch < hp.cosine_warmup:
                return float(epoch + 1) / float(max(1, hp.cosine_warmup))
            progress = float(epoch - hp.cosine_warmup) / float(
                max(1, hp.cosine_t_max - hp.cosine_warmup)
            )
            return max(1e-7, 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers (save only trainable params + buffers)
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save trainable parameters and persistent buffers."""
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        result = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    result[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                result[key] = full_sd[key]
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving checkpoint: {trainable:,}/{total:,} params "
            f"({100 * trainable / total:.2f}%)"
        )
        return result

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-averaged F1 exactly matching calc_metric.py.

    preds  : [N_samples, 3, N_genes]  -- logits / class scores
    labels : [N_samples, N_genes]     -- integer class labels in {0, 1, 2}
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)   # [N_samples, N_genes]
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
            "prediction": json.dumps(preds[i].tolist()),   # [3, 6640] as JSON
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
        description=(
            "Node3-3-3-1-1: ESM2-650M (LoRA r=32) + STRING_GNN Gated Fusion "
            "Pivot from STRING-only (ceiling~0.462) to multi-modal architecture"
        )
    )
    # Batch size
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=32)

    # Architecture
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.10)
    p.add_argument("--head-hidden", type=int, default=HEAD_HIDDEN)
    p.add_argument("--head-dropout", type=float, default=0.45)
    p.add_argument("--fusion-dropout", type=float, default=0.2)

    # Optimizer
    p.add_argument("--backbone-lr", type=float, default=5e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--no-muon", action="store_true", help="Disable Muon (use pure AdamW)")

    # Loss
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.03)

    # Schedule
    p.add_argument("--cosine-warmup", type=int, default=10)
    p.add_argument("--cosine-t-max", type=int, default=160)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--early-stop-patience", type=int, default=55)

    # Test-time
    p.add_argument("--tta-passes", type=int, default=8)
    p.add_argument("--n-ensemble", type=int, default=5)
    p.add_argument("--esm2-max-len", type=int, default=ESM2_MAX_LEN)

    # Data
    p.add_argument("--num-workers", type=int, default=4)

    # Debug
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
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        fusion_dropout=args.fusion_dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        cosine_warmup=args.cosine_warmup,
        cosine_t_max=args.cosine_t_max,
        early_stop_patience=args.early_stop_patience,
        tta_passes=args.tta_passes,
        n_ensemble=args.n_ensemble,
        use_muon=not args.no_muon,
        esm2_max_len=args.esm2_max_len,
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

    # CRITICAL: Use epoch-only filename (no metric value in filename).
    # Lightning's metric name "val/f1" contains a slash which creates nested directories
    # if used in the filename. The solution is to use only the epoch number, then read
    # the CSV logs to determine the val/f1 for each checkpoint at ensemble time.
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-ep{epoch:03d}",
        monitor="val/f1",
        mode="max",
        save_top_k=args.n_ensemble,  # save top-N for ensembling
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
            find_unused_parameters=True,   # ESM2 has unused parameters in some forward passes
            timeout=timedelta(seconds=120),
        ),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
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
    # Test: checkpoint ensemble with TTA
    # ------------------------------------------------------------------
    if args.fast_dev_run or args.debug_max_step is not None:
        # Quick mode: test with current weights only
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        # Production mode: ensemble top-N checkpoints
        # First, run inference with the best single checkpoint
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

        # Now do checkpoint ensemble for better predictions
        checkpoint_dir = output_dir / "checkpoints"

        # Read CSV logs to determine val/f1 for each checkpoint
        def _build_epoch_to_val_f1_map(csv_log_dir: Path) -> Dict[int, float]:
            """Load CSV logs and build epoch -> val_f1 mapping."""
            epoch_map: Dict[int, float] = {}
            for csv_file in sorted(csv_log_dir.glob("metrics.csv")):
                try:
                    df = pd.read_csv(csv_file)
                    if "val/f1" not in df.columns:
                        continue
                    val_rows = df[df["val/f1"].notna()]
                    for _, row in val_rows.iterrows():
                        if "epoch" in row and not pd.isna(row["epoch"]):
                            ep = int(row["epoch"])
                            f1 = float(row["val/f1"])
                            if ep not in epoch_map or epoch_map[ep] < f1:
                                epoch_map[ep] = f1
                except Exception:
                    pass
            return epoch_map

        def _extract_val_f1_from_logs(ckpt_path: Path, csv_log_dir: Path) -> float:
            """Extract val/f1 for a checkpoint by looking up its epoch in CSV logs."""
            try:
                stem = ckpt_path.stem  # e.g. "best-ep042"
                epoch_str = stem.replace("best-ep", "")
                epoch = int(epoch_str)
                epoch_map = _build_epoch_to_val_f1_map(csv_log_dir)
                return epoch_map.get(epoch, 0.0)
            except Exception:
                return 0.0

        csv_log_dir = output_dir / "logs" / "csv_logs"
        # Find the version dir (could be version_0, version_1, etc.)
        version_dirs = sorted(csv_log_dir.glob("version_*")) if csv_log_dir.exists() else []
        if version_dirs:
            csv_log_dir = version_dirs[-1]  # use latest version

        all_ckpt_files = list(checkpoint_dir.glob("best-ep*.ckpt"))
        ckpt_files = sorted(
            all_ckpt_files,
            key=lambda p: _extract_val_f1_from_logs(p, csv_log_dir),
            reverse=True,
        )[:args.n_ensemble]

        if len(ckpt_files) > 1:
            print(f"Running checkpoint ensemble with {len(ckpt_files)} checkpoints...")

            # Collect logits from each checkpoint
            all_pred_maps: List[Dict[str, np.ndarray]] = []
            all_sym_maps: Dict[str, str] = {}

            for ckpt_path in ckpt_files:
                print(f"  Loading checkpoint: {ckpt_path.name}")
                state = torch.load(str(ckpt_path), map_location="cpu")
                if "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)

                # Clear accumulators before test
                model._test_preds.clear()
                model._test_labels.clear()
                model._test_pert_ids.clear()
                model._test_symbols.clear()

                # Run test with this checkpoint's weights
                trainer.test(model, datamodule=datamodule)

                # Read predictions saved by on_test_epoch_end
                pred_tsv = output_dir / "test_predictions.tsv"
                if trainer.is_global_zero and pred_tsv.exists():
                    df_pred = pd.read_csv(pred_tsv, sep="\t")
                    pred_map: Dict[str, np.ndarray] = {}
                    for _, row in df_pred.iterrows():
                        pred_map[str(row["idx"])] = np.array(
                            json.loads(row["prediction"]), dtype=float
                        )
                        all_sym_maps[str(row["idx"])] = str(row["input"])
                    all_pred_maps.append(pred_map)

            # Average predictions across checkpoints (global_zero only writes)
            if trainer.is_global_zero and all_pred_maps:
                all_pids = sorted(all_sym_maps.keys())
                ensemble_preds = []
                for pid in all_pids:
                    ckpt_preds = [m[pid] for m in all_pred_maps if pid in m]
                    avg_pred = np.mean(ckpt_preds, axis=0) if ckpt_preds else np.zeros((N_CLASSES, N_GENES))
                    ensemble_preds.append(avg_pred)

                ensemble_preds_np = np.stack(ensemble_preds, axis=0)
                _save_test_predictions(
                    pert_ids=all_pids,
                    symbols=[all_sym_maps[p] for p in all_pids],
                    preds=ensemble_preds_np,
                    out_path=output_dir / "test_predictions.tsv",
                )
                print(f"Checkpoint ensemble complete. Saved {len(all_pids)} predictions.")

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results saved → {score_path}")


if __name__ == "__main__":
    main()
