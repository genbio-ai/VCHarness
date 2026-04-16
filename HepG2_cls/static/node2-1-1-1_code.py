"""Node 2-1-1-1: ESM2-650M + STRING_GNN with Gated Fusion (Pivot from AIDO.Protein-16B).

Strategy pivot: Abandons the AIDO.Protein-16B direction entirely (node2-1: F1=0.4049,
node2-1-1: catastrophic F1=0.2309) in favour of the ESM2-650M + STRING_GNN gated fusion
approach proven by node4 (F1=0.474).

Key improvements over node4 (the reference implementation, F1=0.474):
  1. Sqrt-inverse class weights (replacing focal loss γ=2): focal loss caused catastrophic
     failures across multiple nodes; sqrt-inverse CE has been consistently stable and is the
     proven loss function for this task.
  2. Label smoothing=0.05: provides additional regularisation that works well with
     imbalanced classes, reduces overconfident predictions.
  3. Higher head dropout (0.3 → 0.4): node4 showed strong overfitting (val F1=0.5005 but
     test F1=0.474). Increasing dropout in the classification head should reduce this gap.
  4. Dual learning rates (backbone=5e-6, head/fusion/GNN=1e-4): node4 used backbone LR=1e-5
     which still allowed overfitting after epoch 18. A lower backbone LR gives the model more
     time to converge before overfitting dominates.
  5. Cosine annealing LR (T_max=100, eta_min=1e-7): replaces ReduceLROnPlateau from node4.
     Smooth decay avoids the plateau-triggered jumps and produces more stable convergence.
     T_max=100 >> expected training duration so no restarts occur.
  6. Intermediate bottleneck in head (1280 → 512 → 256 → 19920 instead of 1280+256 → 19920):
     The 10M-param linear output layer in node4's head was the dominant capacity bottleneck;
     reducing to a 2-stage projection (512-dim hidden) with dropout at each stage provides
     structured regularisation without hurting expressiveness.
  7. Proper checkpoint loading: uses ckpt_path='best' directly (simpler, avoids the manual
     DDP broadcast path that was error-prone in earlier nodes).

Memory budget (ESM2-650M, seq_len=512, bf16, LoRA Q/K/V r=16):
  - ESM2-650M bf16 + LoRA r=16: ~3.5 GB per GPU (from ESM2 skill memory table)
  - STRING_GNN (5.43M, bf16): ~22 MB
  - Head + fusion: ~15 MB
  - Activations (grad ckpt, micro_batch=2): ~4-8 GB
  - Estimated peak: ~12-15 GB per GPU — well within H100 80 GB.

Auxiliary data dependencies:
  - Protein FASTA: /home/data/genome/hg38_gencode_protein.fa (genomic-data-skill)
  - STRING_GNN:    /home/Models/STRING_GNN (string-gnn-model-skill)
  - ESM2-650M:     facebook/esm2_t33_650M_UR50D (HuggingFace, cached locally at
                   ~/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D)
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import pickle
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, EsmForMaskedLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
PROTEIN_FASTA = "/home/data/genome/hg38_gencode_protein.fa"
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")

N_GENES = 6640
N_CLASSES = 3
MAX_SEQ_LEN = 512        # ESM2 max is 1024; 512 covers most proteins and matches node4
ESM2_DIM = 1280          # ESM2-650M hidden size
STRING_DIM = 256         # STRING_GNN output dimension
FALLBACK_SEQ = "M"       # minimal placeholder if ENSG not in FASTA
STRING_FALLBACK_IDX = 18870  # zero-row index for genes absent from STRING


# ---------------------------------------------------------------------------
# Protein sequence lookup helpers
# ---------------------------------------------------------------------------
def _build_ensg_to_seq(fasta_path: str) -> Dict[str, str]:
    """Parse hg38_gencode_protein.fa → ENSG (no version) → longest protein sequence.

    Header format: >ENSP...|ENST...|ENSG00000186092.7|...
    Field index 2 (after '|') contains ENSG with version suffix.
    """
    ensg2seq: Dict[str, str] = {}
    current_ensg: Optional[str] = None
    current_seq_parts: List[str] = []

    def _flush() -> None:
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
class PerturbESM2Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        ensg2seq: Dict[str, str],
        ensg_to_string_idx: Dict[str, int],
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        # Protein sequences
        self.sequences: List[str] = []
        for pid in self.pert_ids:
            ensg = pid.split(".")[0]
            self.sequences.append(ensg2seq.get(ensg, FALLBACK_SEQ))

        # STRING_GNN node indices
        self.string_idxs: List[int] = []
        for pid in self.pert_ids:
            ensg = pid.split(".")[0]
            self.string_idxs.append(ensg_to_string_idx.get(ensg, STRING_FALLBACK_IDX))

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
            "string_idx": self.string_idxs[idx],
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
        micro_batch_size: int = 2,
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
        self.ensg2seq: Optional[Dict[str, str]] = None
        self.ensg_to_string_idx: Optional[Dict[str, int]] = None
        self.train_ds = self.val_ds = self.test_ds = None

    def setup(self, stage: str = "fit") -> None:
        # --- ESM2 Tokenizer: rank-0 downloads, all ranks load ---
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)

        # --- Protein FASTA ---
        self.ensg2seq = get_ensg2seq()

        # --- STRING node names: ENSG_ID → STRING node index ---
        node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        self.ensg_to_string_idx = {name: i for i, name in enumerate(node_names)}

        # --- Datasets ---
        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbESM2Dataset(train_df, self.ensg2seq, self.ensg_to_string_idx)
        self.val_ds = PerturbESM2Dataset(val_df, self.ensg2seq, self.ensg_to_string_idx)
        self.test_ds = PerturbESM2Dataset(test_df, self.ensg2seq, self.ensg_to_string_idx)

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ESM2 tokeniser: standard HF format (no space-separated AAs)
        seqs = [item["seq"][:self.max_seq_len - 2] for item in batch]  # leave room for <cls>/<eos>
        tokenized = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        result = {
            "idx": torch.tensor([item["idx"] for item in batch], dtype=torch.long),
            "pert_id": [item["pert_id"] for item in batch],
            "symbol": [item["symbol"] for item in batch],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "string_idx": torch.tensor([item["string_idx"] for item in batch], dtype=torch.long),
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
# Gated Fusion Module
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Learned scalar-gated fusion between ESM2 protein and STRING_GNN embeddings.

    Uses a lighter scalar gate (projection → 1-dim sigmoid) rather than an
    element-wise gate (projection → ESM2_DIM sigmoid) to reduce parameter count
    and avoid the instability observed in node2-1-1's 5.9M-param gate projection.

    Architecture:
        esm2_proj:   Linear(esm2_dim → fusion_dim)
        string_proj: Linear(string_dim → fusion_dim)
        gate_proj:   Linear(esm2_dim + string_dim → 1)    ← scalar gate

        alpha = sigmoid(gate_proj([esm2, string]))         # scalar ∈ [0,1]
        fused = alpha * esm2_proj(esm2) + (1-alpha) * string_proj(string)
        output: [B, fusion_dim]

    The scalar gate is much more stable than the element-wise gate because:
    - Fewer parameters (2560 → 1 vs 2560 → esm2_dim)
    - Less prone to weight collapse under regularisation
    - Easier to initialise (bias of 0 → alpha=0.5 default)
    """

    def __init__(self, esm2_dim: int, string_dim: int, fusion_dim: int) -> None:
        super().__init__()
        self.esm2_proj = nn.Linear(esm2_dim, fusion_dim)
        self.string_proj = nn.Linear(string_dim, fusion_dim)
        # Scalar gate: projects concatenated [esm2, string] → single logit
        self.gate_proj = nn.Linear(esm2_dim + string_dim, 1)

    def forward(self, esm2_emb: torch.Tensor, string_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            esm2_emb:   [B, esm2_dim]  — ESM2-650M mean-pooled representation
            string_emb: [B, string_dim] — STRING_GNN PPI embedding
        Returns:
            fused: [B, fusion_dim]
        """
        # Project each branch to fusion_dim
        esm2_p = self.esm2_proj(esm2_emb)       # [B, fusion_dim]
        string_p = self.string_proj(string_emb)  # [B, fusion_dim]
        # Scalar gate: single value controls trust in ESM2 vs STRING
        gate_input = torch.cat([esm2_emb, string_emb], dim=-1)  # [B, esm2+string]
        alpha = torch.sigmoid(self.gate_proj(gate_input))        # [B, 1]
        # Weighted combination
        return alpha * esm2_p + (1.0 - alpha) * string_p        # [B, fusion_dim]


# ---------------------------------------------------------------------------
# Prediction Head
# ---------------------------------------------------------------------------
class PerturbHead(nn.Module):
    """Two-stage MLP: [B, in_dim] → [B, 3, N_GENES].

    Using intermediate bottleneck to regularise the large output projection.
    Architecture: LN(in_dim) → Linear(in_dim→hidden) → GELU → Dropout →
                  Linear(hidden→bottleneck) → GELU → Dropout →
                  Linear(bottleneck→n_genes*3) → reshape [B,3,6640]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        bottleneck_dim: int = 256,
        n_genes: int = N_GENES,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, n_genes * N_CLASSES),
        )
        self.n_genes = n_genes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)                           # [B, n_genes * 3]
        return out.view(-1, N_CLASSES, self.n_genes)  # [B, 3, 6640]


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        fusion_dim: int = 512,
        head_hidden_dim: int = 512,
        head_bottleneck_dim: int = 256,
        head_dropout: float = 0.4,
        backbone_lr: float = 5e-6,
        head_lr: float = 1e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = None
        self.gated_fusion: Optional[GatedFusion] = None
        self.head: Optional[PerturbHead] = None

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # ============================================================
        # 1. Pre-compute STRING_GNN frozen embeddings (once per rank)
        # ============================================================
        self.print("Loading STRING_GNN model for one-time embedding computation...")
        string_gnn = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        string_gnn.eval()

        graph_data = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index = graph_data["edge_index"]
        edge_weight = graph_data.get("edge_weight", None)

        with torch.no_grad():
            outputs = string_gnn(edge_index=edge_index, edge_weight=edge_weight)
            string_embs = outputs.last_hidden_state.float().cpu()  # [18870, 256]
        del string_gnn
        self.print(f"STRING_GNN embeddings computed: {string_embs.shape}")

        # Pad with zero row at index 18870 for "not-in-STRING" genes
        pad = torch.zeros(1, STRING_DIM, dtype=torch.float32)
        string_embs_padded = torch.cat([string_embs, pad], dim=0)  # [18871, 256]
        self.register_buffer("string_emb_table", string_embs_padded)

        # ============================================================
        # 2. Load ESM2-650M with LoRA (r=16)
        # ============================================================
        self.print(f"Loading ESM2-650M with LoRA r={self.hparams.lora_r}...")
        backbone = EsmForMaskedLM.from_pretrained(
            ESM2_MODEL_NAME, dtype=torch.bfloat16
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            target_modules=["query", "key", "value"],
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable()
        if hasattr(self.backbone, "config"):
            self.backbone.config.use_cache = False

        # Cast trainable (LoRA) params to float32 for stable optimisation
        for _name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ============================================================
        # 3. Gated fusion + prediction head
        # ============================================================
        # Scalar-gated fusion: ESM2 (1280) + STRING (256) → fusion_dim
        self.gated_fusion = GatedFusion(
            esm2_dim=ESM2_DIM,
            string_dim=STRING_DIM,
            fusion_dim=self.hparams.fusion_dim,
        )

        # Two-stage bottleneck head: fusion_dim → hidden → bottleneck → 19920
        self.head = PerturbHead(
            in_dim=self.hparams.fusion_dim,
            hidden_dim=self.hparams.head_hidden_dim,
            bottleneck_dim=self.hparams.head_bottleneck_dim,
            dropout=self.hparams.head_dropout,
        )

        # ============================================================
        # 4. Loss: CrossEntropyLoss + sqrt-inverse class weights + label smoothing
        #
        # Focal loss (used in node4) caused instability. sqrt-inverse CE has been
        # consistently stable across all successful nodes in the tree.
        # ============================================================
        freq = torch.tensor([0.9282, 0.0477, 0.0241], dtype=torch.float32)
        weights = torch.sqrt(1.0 / freq)
        weights = weights / weights.mean()  # normalise mean weight = 1.0
        # weights ≈ [0.26, 1.14, 1.60] — max ratio ~6.2×
        self.register_buffer("class_weights", weights)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"ESM2-650M LoRA(r={self.hparams.lora_r}) + STRING_GNN Scalar-Gated Fusion | "
            f"trainable={trainable:,}/{total:,} params "
            f"({100.0 * trainable / total:.2f}%)"
        )

    def _encode(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Encode a batch → [B, fusion_dim] via scalar gated fusion."""
        # --- ESM2-650M forward ---
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        # Last hidden state: [B, T, 1280]
        hidden = out.hidden_states[-1].float()

        # Mean pooling over valid (non-special, non-padding) tokens
        # ESM2 special token ids: CLS=0, EOS=2, PAD=1
        input_ids = batch["input_ids"]
        special_mask = (
            (input_ids == 0) | (input_ids == 1) | (input_ids == 2)
        )  # [B, T] — True for special/pad tokens
        valid_mask = ~special_mask  # [B, T] — True for amino acid tokens

        # Safe mean pooling
        valid_mask_f = valid_mask.float().unsqueeze(-1)  # [B, T, 1]
        sum_emb = (hidden * valid_mask_f).sum(dim=1)      # [B, 1280]
        count = valid_mask_f.sum(dim=1).clamp(min=1e-9)  # [B, 1]
        esm2_emb = sum_emb / count                         # [B, 1280]

        # --- STRING_GNN embedding lookup → [B, 256] ---
        string_emb = self.string_emb_table[batch["string_idx"]].float()  # [B, 256]

        # --- Scalar gated fusion → [B, fusion_dim] ---
        return self.gated_fusion(esm2_emb, string_emb)  # [B, fusion_dim]

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """CrossEntropyLoss with sqrt-inverse weights + label smoothing=0.05."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=0.05,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        emb = self._encode(batch)
        logits = self.head(emb)
        loss = self._compute_loss(logits, batch["label"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        emb = self._encode(batch)
        logits = self.head(emb)
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

        all_preds = self.all_gather(preds_local)
        all_labels = self.all_gather(labels_local)
        if self.trainer.world_size > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
            all_labels = all_labels.view(-1, N_GENES)

        if self.trainer.is_global_zero:
            f1 = _compute_per_gene_f1(
                all_preds.float().cpu().numpy(), all_labels.cpu().numpy()
            )
            self.log("val_f1", f1, prog_bar=True, sync_dist=False)
        else:
            self.log("val_f1", 0.0, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        emb = self._encode(batch)
        logits = self.head(emb)  # [B, 3, 6640]
        self._test_preds.append(logits.detach().cpu().float())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist

        preds_local = torch.cat(self._test_preds, dim=0)
        self._test_preds.clear()

        all_preds = self.all_gather(preds_local)
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)

        world_size = self.trainer.world_size
        local_ids = list(self._test_pert_ids)
        local_syms = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        all_pert_ids: List[str] = []
        all_symbols: List[str] = []

        if world_size > 1:
            local_ids_bytes = pickle.dumps(local_ids)
            local_syms_bytes = pickle.dumps(local_syms)
            local_ids_len = len(local_ids_bytes)
            local_syms_len = len(local_syms_bytes)

            ids_lens = [0] * world_size
            syms_lens = [0] * world_size
            dist.all_gather_object(ids_lens, local_ids_len)
            dist.all_gather_object(syms_lens, local_syms_len)

            max_ids_len = max(ids_lens) if ids_lens else 0
            max_syms_len = max(syms_lens) if syms_lens else 0

            if max_ids_len > 0:
                ids_np = np.frombuffer(local_ids_bytes, dtype=np.uint8).copy()
                ids_tensor = F.pad(
                    torch.from_numpy(ids_np).cuda(), (0, max_ids_len - local_ids_len)
                )
                gathered_ids = [
                    torch.zeros(max_ids_len, dtype=torch.uint8, device="cuda")
                    for _ in range(world_size)
                ]
                dist.all_gather(gathered_ids, ids_tensor)
                for r, blen in enumerate(ids_lens):
                    if blen > 0:
                        all_pert_ids.extend(
                            pickle.loads(gathered_ids[r][:blen].cpu().numpy().tobytes())
                        )

            if max_syms_len > 0:
                syms_np = np.frombuffer(local_syms_bytes, dtype=np.uint8).copy()
                syms_tensor = F.pad(
                    torch.from_numpy(syms_np).cuda(), (0, max_syms_len - local_syms_len)
                )
                gathered_syms = [
                    torch.zeros(max_syms_len, dtype=torch.uint8, device="cuda")
                    for _ in range(world_size)
                ]
                dist.all_gather(gathered_syms, syms_tensor)
                for r, blen in enumerate(syms_lens):
                    if blen > 0:
                        all_symbols.extend(
                            pickle.loads(gathered_syms[r][:blen].cpu().numpy().tobytes())
                        )
        else:
            all_pert_ids = local_ids
            all_symbols = local_syms

        # Gather test labels if available
        has_labels = bool(self._test_labels)
        all_labels = None
        if has_labels:
            labels_local = torch.cat(self._test_labels, dim=0)
            self._test_labels.clear()
            all_labels_gathered = self.all_gather(labels_local)
            all_labels = all_labels_gathered.view(-1, N_GENES)

        if self.trainer.is_global_zero:
            n_preds = all_preds.shape[0]
            n_ids = len(all_pert_ids)
            min_len = min(n_preds, n_ids)

            _save_test_predictions(
                pert_ids=all_pert_ids[:min_len],
                symbols=all_symbols[:min_len],
                preds=all_preds[:min_len].float().cpu().numpy(),
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

            if has_labels and all_labels is not None:
                f1 = _compute_per_gene_f1(
                    all_preds[:min_len].float().cpu().numpy(),
                    all_labels[:min_len].cpu().numpy(),
                )
                self.log("test_f1", f1, prog_bar=True, sync_dist=False)
                self.print(f"Test F1: {f1:.4f}")

    def configure_optimizers(self):
        # Two parameter groups: backbone LoRA (low LR) vs everything else (high LR)
        backbone_params = [p for n, p in self.backbone.named_parameters() if p.requires_grad]
        other_params = (
            list(self.gated_fusion.parameters()) +
            list(self.head.parameters())
        )
        # STRING embedding table is a frozen buffer (no grad), not in optimizer

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.backbone_lr},
                {"params": other_params, "lr": self.hparams.head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
        )
        # Cosine annealing over T_max=100 epochs (much larger than expected training duration,
        # so no cosine restart occurs within normal training)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-7,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and persistent buffers."""
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys = {prefix + n for n, _ in self.named_buffers()}
        save_keys = trainable_keys | buffer_keys
        result = {k: v for k, v in full_sd.items() if k in save_keys}

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buf_total = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable}/{total} params "
            f"({100.0 * trainable / total:.2f}%), plus {buf_total} buffer values"
        )
        return result

    def load_state_dict(self, state_dict, strict=True):
        """Load partial checkpoint (trainable params + buffers only)."""
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro F1, averaged over all n_genes (matches calc_metric.py).

    Args:
        preds: [n_samples, 3, n_genes] — class logits / probabilities
        labels: [n_samples, n_genes] — integer class labels in {0, 1, 2}
    Returns:
        scalar F1
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [n_samples, n_genes]
    n_genes = labels.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = labels[:, g].flatten().astype(np.int32)
        yh = y_hat[:, g].flatten().astype(np.int32)
        per_class = sk_f1(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        f1_vals.append(float(per_class[present].mean()) if present.any() else 0.0)
    return float(np.mean(f1_vals))


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
    p = argparse.ArgumentParser(
        description="Node2-1-1-1: ESM2-650M LoRA + STRING_GNN Scalar-Gated Fusion"
    )
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--global-batch-size", type=int, default=16)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--backbone-lr", type=float, default=5e-6)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--fusion-dim", type=int, default=512)
    p.add_argument("--head-hidden-dim", type=int, default=512)
    p.add_argument("--head-bottleneck-dim", type=int, default=256)
    p.add_argument("--head-dropout", type=float, default=0.4)
    p.add_argument("--early-stop-patience", type=int, default=10)
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

    model = PerturbModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        fusion_dim=args.fusion_dim,
        head_hidden_dim=args.head_hidden_dim,
        head_bottleneck_dim=args.head_bottleneck_dim,
        head_dropout=args.head_dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
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
        filename="best-ep={epoch:03d}",
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
        strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=600)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=1.0,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        gradient_clip_val=1.0,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        # Use the explicit best checkpoint path (broadcast to all DDP ranks after training)
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_cb.best_model_path)


if __name__ == "__main__":
    main()
