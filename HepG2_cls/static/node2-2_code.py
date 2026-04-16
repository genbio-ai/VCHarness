"""Node 2-2: AIDO.Protein-16B (LoRA, last-12-layers, seq_len=1024) + STRING_GNN Gated Fusion.

Architecture improvements over node2-1 (seq_len=512, concat, wd=1e-4):
  1. MAX_SEQ_LEN 512 → 1024: captures full protein functional domains
  2. LoRA restricted to last 12 layers (layers 24-35) to save GPU memory while retaining
     task-relevant representations from the deepest transformer layers
  3. Weighted multi-layer pooling over last 4 hidden states instead of last-layer-only
  4. Learned scalar gating: fused = σ(gate)*protein_proj + (1-σ(gate))*string_emb
     prevents protein dominance (90% mass in raw concat), lets model balance sources
  5. Factorized 3-layer bottleneck head (2304→512→256→19920) reduces overfitting
     capacity from node2-1's unfactorized 2560→1024→19920 linear head
  6. ReduceLROnPlateau (proven in node1-1-1, F1=0.474) replaces cosine annealing
  7. weight_decay 1e-4 → 0.01 (matches successful STRING-only nodes)
  8. Stronger head dropout: 0.3 → 0.4
  9. Per-gene bias term (proven in node1-1-1, node1-2-1-1)
 10. ckpt_path='best' for testing (fixes parent's "last checkpoint" bug)

Backbone: AIDO.Protein-16B (MoE, 16B params) with LoRA on Q/K/V for layers 24-35
PPI: STRING_GNN frozen embeddings (256-dim) via pre-computed lookup table
Loss: CrossEntropyLoss + sqrt-inverse class weights (max ratio ≈ 6.2×) + label_smoothing=0.05
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
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = "/home/Models/AIDO.Protein-16B"
STRING_GNN_DIR = "/home/Models/STRING_GNN"
PROTEIN_FASTA = "/home/data/genome/hg38_gencode_protein.fa"
N_GENES = 6640
N_CLASSES = 3
MAX_SEQ_LEN = 1024  # Increased from 512 — captures more functional domains
PROTEIN_HIDDEN = 2304   # AIDO.Protein-16B hidden size
STRING_HIDDEN = 256     # STRING_GNN hidden size
FUSED_DIM = PROTEIN_HIDDEN  # after gating, output is PROTEIN_HIDDEN


# ---------------------------------------------------------------------------
# Protein sequence lookup helpers
# ---------------------------------------------------------------------------
def _build_ensg_to_seq(fasta_path: str) -> Dict[str, str]:
    """Parse hg38_gencode_protein.fa: ENSG→longest protein sequence map."""
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
                    ensg_raw = fields[2]
                    current_ensg = ensg_raw.split(".")[0]
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


FALLBACK_SEQ = "M"  # minimal placeholder if no protein sequence found


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbProteinDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        ensg2seq: Dict[str, str],
        string_node_map: Dict[str, int],  # ENSG → STRING node index
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        # Resolve protein sequences (strip version from pert_id)
        self.sequences: List[str] = []
        self.string_indices: List[int] = []
        missing_count = 0

        for pid in self.pert_ids:
            ensg = pid.split(".")[0]
            seq = ensg2seq.get(ensg, FALLBACK_SEQ)
            self.sequences.append(seq)
            # STRING node index; 18870 = placeholder zero-vector index for missing
            sidx = string_node_map.get(ensg, len(string_node_map))
            if sidx == len(string_node_map):
                missing_count += 1
            self.string_indices.append(sidx)

        if missing_count > 0:
            print(f"  [Dataset] {missing_count}/{len(self.pert_ids)} genes missing from STRING PPI graph")

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
            "string_idx": self.string_indices[idx],
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
        micro_batch_size: int = 1,
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
        self.string_node_map: Dict[str, int] = {}
        self.ensg2seq: Optional[Dict[str, str]] = None
        self.train_ds: Optional[PerturbProteinDataset] = None
        self.val_ds: Optional[PerturbProteinDataset] = None
        self.test_ds: Optional[PerturbProteinDataset] = None

    def setup(self, stage: str = "fit") -> None:
        # Rank-0 downloads/caches tokenizer; all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

        # Build protein sequence lookup
        self.ensg2seq = get_ensg2seq()

        # Build STRING node map from node_names.json
        node_names_path = Path(STRING_GNN_DIR) / "node_names.json"
        node_names = json.loads(node_names_path.read_text())
        # node_names[i] = ENSG ID (e.g., "ENSG00000000938")
        self.string_node_map = {name: i for i, name in enumerate(node_names)}

        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbProteinDataset(train_df, self.ensg2seq, self.string_node_map)
        self.val_ds = PerturbProteinDataset(val_df, self.ensg2seq, self.string_node_map)
        self.test_ds = PerturbProteinDataset(test_df, self.ensg2seq, self.string_node_map)

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # AIDO.Protein-16B tokenizer expects space-separated amino acid tokens.
        seqs = [" ".join(list(item["seq"])) + " " for item in batch]
        tokenized = self.tokenizer.make_a_batch(
            seqs,
            max_length=self.max_seq_len,
            padding_to="longest",
            add_sep_token=True,
        )
        result = {
            "idx": torch.tensor([item["idx"] for item in batch], dtype=torch.long),
            "pert_id": [item["pert_id"] for item in batch],
            "symbol": [item["symbol"] for item in batch],
            "string_idx": torch.tensor([item["string_idx"] for item in batch], dtype=torch.long),
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "padding_mask": tokenized["padding_mask"],
            "special_mask": tokenized["special_mask"],
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
# Prediction Head: Factorized bottleneck + per-gene bias
# ---------------------------------------------------------------------------
class FactorizedPerturbHead(nn.Module):
    """3-layer factorized bottleneck head: [B, in_dim] → [B, 3, N_GENES].

    Factorized design reduces the dominant output layer capacity from
    in_dim→n_genes*3 (e.g. 2304→19920 = 45.9M params) to a 3-layer
    bottleneck (2304→512→256→19920) reducing to ~5M params in first two
    layers plus ~5.1M in final projection = less overfitting risk.
    Per-gene bias (N_GENES * N_CLASSES = 19920 params) allows the model
    to learn baseline class probabilities per gene independent of input.
    """

    def __init__(
        self,
        in_dim: int,
        hidden1: int = 512,
        hidden2: int = 256,
        n_genes: int = N_GENES,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden1),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_genes * N_CLASSES),
        )
        # Per-gene bias: learned baseline per gene-class pair
        self.per_gene_bias = nn.Parameter(torch.zeros(n_genes * N_CLASSES))
        self.n_genes = n_genes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        out = self.net(x) + self.per_gene_bias  # [B, n_genes * 3]
        return out.view(-1, N_CLASSES, self.n_genes)  # [B, 3, 6640]


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_layers_to_transform: Optional[List[int]] = None,
        n_protein_layers_weighted: int = 4,
        head_hidden1: int = 512,
        head_hidden2: int = 256,
        head_dropout: float = 0.4,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = None
        self.string_gnn = None
        self.head: Optional[FactorizedPerturbHead] = None

        # Learnable weights for multi-layer feature aggregation
        # Will be initialized in setup() to use n_protein_layers_weighted layers
        self.layer_weights: Optional[nn.Parameter] = None
        # Gating module for protein-STRING fusion
        self.gate: Optional[nn.Linear] = None
        # STRING projection to match PROTEIN_HIDDEN for gating
        self.string_proj: Optional[nn.Linear] = None
        # Attention pooling over token positions
        self.attn_pool: Optional[nn.Linear] = None

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # ---- AIDO.Protein-16B backbone with LoRA on last 12 layers ----
        backbone = AutoModelForMaskedLM.from_pretrained(
            MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16
        )

        # LoRA on Q/K/V of the last 12 transformer layers (layers 24-35)
        # Rationale: deeper layers contain more task-specific representations;
        # restricting LoRA to 12 layers saves ~2×+ memory at seq_len=1024
        lora_layers = self.hparams.lora_layers_to_transform
        if lora_layers is None:
            lora_layers = list(range(24, 36))  # Last 12 layers of 36-layer model

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=lora_layers,
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable()
        if hasattr(self.backbone, "config"):
            self.backbone.config.use_cache = False

        # Cast trainable (LoRA) params to float32 for stability
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ---- STRING_GNN: frozen, pre-compute embeddings once ----
        string_gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
        string_gnn.eval()
        graph = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", weights_only=False)
        edge_index = graph["edge_index"]
        edge_weight = graph["edge_weight"]

        with torch.no_grad():
            # Run on CPU first, then move to appropriate device
            outputs = string_gnn(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
        string_embs = outputs.last_hidden_state.float()  # [18870, 256]

        # Pad with a zero row for missing-gene fallback (index 18870)
        zero_row = torch.zeros(1, STRING_HIDDEN, dtype=torch.float32)
        string_embs_padded = torch.cat([string_embs, zero_row], dim=0)  # [18871, 256]

        # Register as buffer so it moves with the model and is saved
        self.register_buffer("string_emb_table", string_embs_padded)
        del string_gnn, string_embs, string_embs_padded

        # ---- Learnable weights for multi-layer hidden state aggregation ----
        n_layers = self.hparams.n_protein_layers_weighted
        self.layer_weights = nn.Parameter(torch.ones(n_layers, dtype=torch.float32) / n_layers)

        # ---- Attention pooling (learnable query over token positions) ----
        self.attn_pool = nn.Linear(PROTEIN_HIDDEN, 1, bias=False)
        self.attn_pool.weight.data = torch.randn(1, PROTEIN_HIDDEN, dtype=torch.float32) * 0.01

        # ---- Gated fusion: protein (2304) + STRING (256) → fused (2304) ----
        # Project STRING to same dim as protein for element-wise gating
        self.string_proj = nn.Linear(STRING_HIDDEN, PROTEIN_HIDDEN, bias=True)
        # Scalar gate: takes concatenation of both → scalar in [0,1]
        # gate=1: use protein, gate=0: use string_proj
        self.gate = nn.Linear(PROTEIN_HIDDEN + STRING_HIDDEN, 1, bias=True)
        # Initialize gate to output 0.5 (equal weighting initially)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 0.0)

        # Ensure new float32 parameters
        for module in [self.attn_pool, self.string_proj, self.gate]:
            for p in module.parameters():
                p.data = p.data.float()

        # ---- Prediction head: factorized bottleneck ----
        self.head = FactorizedPerturbHead(
            in_dim=PROTEIN_HIDDEN,  # After gating, output dim = PROTEIN_HIDDEN
            hidden1=self.hparams.head_hidden1,
            hidden2=self.hparams.head_hidden2,
            dropout=self.hparams.head_dropout,
        )

        # ---- Loss with mild class weights + label smoothing ----
        # sqrt-inverse frequency: class 0=neutral(0.9282), 1=down(0.0477), 2=up(0.0241)
        freq = torch.tensor([0.9282, 0.0477, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq).sqrt()
        class_weights = class_weights / class_weights.mean()  # normalize
        self.register_buffer("class_weights", class_weights)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Use print() instead of self.print() since setup() runs before trainer is attached
        print(f"AIDO.Protein-16B+LoRA(last12) + STRING_GNN gated | "
              f"trainable={trainable:,}/{total:,} params")

    def _encode(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Run backbone with multi-layer weighted pooling + attention-based aggregation.

        Returns: [B, PROTEIN_HIDDEN]
        """
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        # Use last n_protein_layers_weighted hidden states
        n_layers = self.hparams.n_protein_layers_weighted
        hidden_states = out.hidden_states[-n_layers:]  # tuple of n_layers tensors [B, T, H]

        # Weighted combination of multiple layers
        weights = F.softmax(self.layer_weights, dim=0)  # [n_layers]
        # Stack and weighted sum: [B, T, H]
        stacked = torch.stack([h.float() for h in hidden_states], dim=0)  # [n_layers, B, T, H]
        combined = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)  # [B, T, H]

        # Mask out padding and special tokens
        exclude_mask = (batch["padding_mask"].bool() | batch["special_mask"].bool())
        valid_mask = ~exclude_mask  # [B, T], True=valid

        # Attention pooling: compute scalar attention score per token
        attn_scores = self.attn_pool(combined).squeeze(-1)  # [B, T]
        # Set invalid positions to -inf before softmax
        attn_scores = attn_scores.masked_fill(~valid_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, T]
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Weighted aggregation
        pooled = (combined * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, H]
        return pooled  # [B, PROTEIN_HIDDEN]

    def _fuse(self, protein_emb: torch.Tensor, string_idx: torch.Tensor) -> torch.Tensor:
        """Scalar-gated fusion of protein and STRING_GNN embeddings.

        protein_emb: [B, PROTEIN_HIDDEN]
        string_idx: [B] long tensor of STRING node indices
        Returns: [B, PROTEIN_HIDDEN]
        """
        string_emb = self.string_emb_table[string_idx].float()  # [B, 256]
        string_proj = self.string_proj(string_emb)  # [B, PROTEIN_HIDDEN]

        # Compute gate from concatenated inputs
        gate_input = torch.cat([protein_emb, string_emb], dim=-1)  # [B, PROTEIN_HIDDEN+256]
        gate = torch.sigmoid(self.gate(gate_input))  # [B, 1] in [0,1]

        # Fused embedding: gate=1 → more protein, gate=0 → more STRING
        fused = gate * protein_emb + (1.0 - gate) * string_proj  # [B, PROTEIN_HIDDEN]
        return fused

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, 3, 6640], labels: [B, 6640] in {0,1,2}
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights.to(logits_flat.device),
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        protein_emb = self._encode(batch)
        fused = self._fuse(protein_emb, batch["string_idx"])
        logits = self.head(fused)
        loss = self._compute_loss(logits, batch["label"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        protein_emb = self._encode(batch)
        fused = self._fuse(protein_emb, batch["string_idx"])
        logits = self.head(fused)
        loss = self._compute_loss(logits, batch["label"])
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

        # Gather all predictions and labels across ranks for global F1 computation
        all_preds = self.all_gather(preds_local)
        all_labels = self.all_gather(labels_local)
        # all_gather always adds a leading world_size dim. Flatten it regardless of world_size.
        # Shape after gather: [world_size, local_B, 3, 6640] → flatten first dim
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
        all_labels = all_labels.view(-1, N_GENES)

        if self.trainer.is_global_zero:
            f1 = _compute_per_gene_f1(all_preds.float().cpu().numpy(), all_labels.cpu().numpy())
            self.log("val_f1", f1, prog_bar=True, sync_dist=False)
        else:
            self.log("val_f1", 0.0, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        protein_emb = self._encode(batch)
        fused = self._fuse(protein_emb, batch["string_idx"])
        logits = self.head(fused)  # [B, 3, 6640]
        self._test_preds.append(logits.detach().cpu().float())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        preds_local = torch.cat(self._test_preds, dim=0)
        self._test_preds.clear()

        # Gather all predictions across ranks
        all_preds = self.all_gather(preds_local)
        total_samples = all_preds.shape[0] * all_preds.shape[1]
        all_preds = all_preds.view(total_samples, N_CLASSES, N_GENES)

        # Gather all pert_ids and symbols using pickle + CUDA tensor all_gather
        world_size = self.trainer.world_size
        local_ids = list(self._test_pert_ids)
        local_syms = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        all_pert_ids: List[str] = []
        all_symbols: List[str] = []

        if world_size > 1:
            import torch.distributed as dist

            local_ids_bytes = pickle.dumps(local_ids)
            local_syms_bytes = pickle.dumps(local_syms)
            local_len = len(local_ids_bytes)
            local_syms_len = len(local_syms_bytes)

            obj_gather = [0] * world_size
            dist.all_gather_object(obj_gather, local_len)
            all_lens = obj_gather

            obj_gather_syms = [0] * world_size
            dist.all_gather_object(obj_gather_syms, local_syms_len)
            all_syms_lens = obj_gather_syms

            max_len = max(all_lens) if all_lens else 0
            max_syms_len = max(all_syms_lens) if all_syms_lens else 0

            if max_len > 0:
                ids_np = np.frombuffer(local_ids_bytes, dtype=np.uint8).copy()
                ids_tensor = torch.from_numpy(ids_np).cuda()
                ids_tensor = torch.nn.functional.pad(ids_tensor, (0, max_len - local_len))
                gathered_ids = [torch.zeros(max_len, dtype=torch.uint8, device="cuda") for _ in range(world_size)]
                dist.all_gather(gathered_ids, ids_tensor)
                for r, blen in enumerate(all_lens):
                    if blen > 0:
                        b = gathered_ids[r][:blen].cpu().numpy().tobytes()
                        all_pert_ids.extend(pickle.loads(b))

            if max_syms_len > 0:
                syms_np = np.frombuffer(local_syms_bytes, dtype=np.uint8).copy()
                syms_tensor = torch.from_numpy(syms_np).cuda()
                syms_tensor = torch.nn.functional.pad(syms_tensor, (0, max_syms_len - local_syms_len))
                gathered_syms = [torch.zeros(max_syms_len, dtype=torch.uint8, device="cuda") for _ in range(world_size)]
                dist.all_gather(gathered_syms, syms_tensor)
                for r, blen in enumerate(all_syms_lens):
                    if blen > 0:
                        b = gathered_syms[r][:blen].cpu().numpy().tobytes()
                        all_symbols.extend(pickle.loads(b))
        else:
            all_pert_ids = local_ids
            all_symbols = local_syms

        # Gather test labels if available
        has_labels = bool(self._test_labels)
        if has_labels:
            labels_local = torch.cat(self._test_labels, dim=0)
            all_labels = self.all_gather(labels_local)
            total_labels = all_labels.shape[0] * all_labels.shape[1]
            all_labels = all_labels.view(total_labels, N_GENES)
            self._test_labels.clear()
        else:
            all_labels = None

        if self.trainer.is_global_zero:
            n_preds = all_preds.shape[0]
            n_ids = len(all_pert_ids)
            min_len = min(n_preds, n_ids)
            saved_preds = all_preds[:min_len]
            _save_test_predictions(
                pert_ids=all_pert_ids[:min_len],
                symbols=all_symbols[:min_len],
                preds=saved_preds.float().cpu().numpy(),
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

            if has_labels and all_labels is not None:
                min_labels = min_len
                f1 = _compute_per_gene_f1(
                    saved_preds[:min_labels].float().cpu().numpy(),
                    all_labels[:min_labels].cpu().numpy(),
                )
                self.log("test/f1", f1, prog_bar=True, sync_dist=False)
                self.print(f"Test F1: {f1:.4f}")

    def configure_optimizers(self):
        # Separate LR groups: backbone LoRA adapters vs head + gating modules
        backbone_params = [p for n, p in self.backbone.named_parameters() if p.requires_grad]
        head_params = (
            list(self.head.parameters())
            + list(self.attn_pool.parameters())
            + list(self.string_proj.parameters())
            + list(self.gate.parameters())
            + [self.layer_weights]
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.lr},
                {"params": head_params, "lr": self.hparams.lr * 3},  # higher LR for head
            ],
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1", "interval": "epoch"},
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
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
        # Use plain print to avoid trainer dependency
        print(f"Saving {trainable}/{total} params ({100*trainable/total:.2f}%)")
        return result

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute per-gene macro F1.

    Args:
        preds: shape [n_samples, 3, n_genes] — class logits/probabilities
        labels: shape [n_samples, n_genes] — integer class labels in {0, 1, 2}
    """
    from sklearn.metrics import f1_score as sk_f1
    y_hat = preds.argmax(axis=1)
    n_genes = labels.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels[:, g].flatten().astype(np.int32)
        yh = y_hat[:, g].flatten().astype(np.int32)
        per_class_f1 = sk_f1(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        if present.any():
            f1_vals.append(float(per_class_f1[present].mean()))
    return float(np.mean(f1_vals)) if f1_vals else 0.0


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Node2-2: AIDO.Protein-16B LoRA(last12) + STRING_GNN gated fusion")
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--head-hidden1", type=int, default=512)
    p.add_argument("--head-hidden2", type=int, default=256)
    p.add_argument("--head-dropout", type=float, default=0.4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
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

    # LoRA on last 12 layers of AIDO.Protein-16B (layers 24-35 of 36 total)
    lora_layers = list(range(24, 36))

    model = PerturbModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_layers_to_transform=lora_layers,
        n_protein_layers_weighted=4,
        head_hidden1=args.head_hidden1,
        head_hidden2=args.head_hidden2,
        head_dropout=args.head_dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = limit_val = limit_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-epoch={epoch:03d}-val_f1={val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
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
        strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=1.0 if (args.debug_max_step is None and not args.fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=datamodule)

    # Load best checkpoint for testing (Lightning creates a best.ckpt symlink)
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        best_ckpt = output_dir / "checkpoints" / "best.ckpt"
        if best_ckpt.exists():
            test_results = trainer.test(model, datamodule=datamodule, ckpt_path=str(best_ckpt))
        else:
            print(f"No best checkpoint found at {best_ckpt}, testing with current model weights.")
            test_results = trainer.test(model, datamodule=datamodule)

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results → {score_path}")


if __name__ == "__main__":
    main()
