#!/usr/bin/env python3
"""
Node: AIDO.Cell-10M Minimal LoRA + STRING_GNN Learnable Adapter + Compact Head
===============================================================================
Key innovations over parent (node1-2-1) and sibling (node1-2):

Parent failure: 11.6M trainable params (7733/sample) → severe overfitting, test F1=0.411
Sibling failure: Fully frozen backbones → representational ceiling at ~0.41 F1

This node finds the sweet spot between adaptation and regularization:

  1. AIDO.Cell-10M with ULTRA-MINIMAL LoRA (r=4, last 2 of 8 layers, ~12K params)
     - Distinct from parent (100M, r=16, 12 layers, 740K params)
     - Distinct from sibling (fully frozen, 0 backbone trainable params)
     - Allows backbone to adapt attention patterns to synthetic one-hot inputs
       without the overfitting risk of larger LoRA

  2. STRING_GNN (frozen) + LEARNABLE LINEAR ADAPTER (256→128, ~33K params)
     - Distinct from sibling (STRING_GNN completely frozen, no adapter)
     - The adapter learns which PPI dimensions predict differential expression
     - Learnable null_raw embedding for genes not in STRING vocab (256 params)

  3. Very compact low-rank head: Linear(640→32) → GELU → Dropout(0.6) → Linear(32→19920)
     - ~679K head params (vs 10.87M parent, vs 1.33M sibling)
     - Very strong dropout (0.6) for regularization

  4. Total trainable: ~724K params → ~483 params/sample (vs 7733 parent, 897 sibling)

  5. Two optimizer param groups:
     - Backbone LoRA: lr=5e-5, wd=5e-2 (very conservative)
     - STRING adapter + head: lr=5e-4, wd=5e-2 (standard)

  6. Focal loss (γ=3.0), class_weights=[2.0,1.0,4.0], label_smoothing=0.10

  7. Longer training: max_epochs=120, patience=20 (minimal LoRA needs more epochs)

Architecture:
  AIDO.Cell-10M + LoRA → dual pool → [B, 512]
  STRING_GNN (frozen) → lookup → [B, 256] + learnable null_raw → adapter → [B, 128]
  cat([512, 128]) → [B, 640] → LayerNorm → Linear(32) → GELU → Dropout → Linear(19920)
  → [B, 3, 6640]
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
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
N_GENES = 6_640
N_CLASSES = 3

AIDO_MODEL_PATH = "/home/Models/AIDO.Cell-10M"   # 10M variant, 256-dim hidden
STRING_GNN_PATH = "/home/Models/STRING_GNN"

AIDO_DIM = 256              # AIDO.Cell-10M hidden dim
STRING_DIM = 256            # STRING_GNN output dim
ADAPTED_STRING_DIM = 128    # after learnable adapter
DUAL_POOL_DIM = AIDO_DIM * 2  # 512 (gene_pos + mean_pool)
FUSION_DIM = DUAL_POOL_DIM + ADAPTED_STRING_DIM  # 640

# Moderate minority boost (softer than parent's [5.0, 1.0, 10.0])
# Train distribution: ~3.41% down (-1→0), ~95.48% unchanged (0→1), ~1.10% up (+1→2)
CLASS_WEIGHTS = torch.tensor([2.0, 1.0, 4.0], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STRING_GNN Embedding Extraction (run once, frozen after)
# ─────────────────────────────────────────────────────────────────────────────
def extract_string_gnn_embeddings(model_path: str = STRING_GNN_PATH) -> tuple:
    """
    Run STRING_GNN once on CPU to extract frozen node embeddings.
    Returns:
        emb_matrix: [18870, 256] float32 tensor on CPU
        node_name_to_idx: dict mapping Ensembl ID → row index in emb_matrix
    """
    model_dir = Path(model_path)
    node_names = json.loads((model_dir / "node_names.json").read_text())
    node_name_to_idx = {name: i for i, name in enumerate(node_names)}

    gnn_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    gnn_model.eval()

    graph = torch.load(str(model_dir / "graph_data.pt"), map_location="cpu")
    edge_index = graph["edge_index"]
    edge_weight = graph["edge_weight"]

    with torch.no_grad():
        outputs = gnn_model(edge_index=edge_index, edge_weight=edge_weight)

    emb_matrix = outputs.last_hidden_state.detach().cpu().float()  # [18870, 256]
    del gnn_model
    torch.cuda.empty_cache()

    return emb_matrix, node_name_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with class weighting and label smoothing."""

    def __init__(self, gamma: float = 3.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: [N, C], targets: [N] int64."""
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # Focal weighting: detach to avoid double-differentiating through CE
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none").detach())
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper (mirrors calc_metric.py logic exactly)
# ─────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """
    y_pred:          [N, 3, G] float  (probabilities or logits)
    y_true_remapped: [N, G]    int    ({0, 1, 2} after +1 remap)
    Returns: macro F1 averaged over G genes.
    """
    n_genes = y_true_remapped.shape[1]
    f1_vals: List[float] = []
    y_hat = y_pred.argmax(axis=1)  # [N, G]
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp = y_hat[:, g]
        present = np.array([(yt == c).any() for c in range(3)])
        pf1 = sk_f1_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """Dataset holding pre-computed STRING_GNN embeddings + AIDO.Cell tokenized inputs."""

    def __init__(
        self,
        df: pd.DataFrame,
        string_embs: torch.Tensor,      # [N, 256] float32 (zeros for unknown genes)
        string_valid: torch.Tensor,     # [N] bool (True if gene found in STRING vocab)
        input_ids: torch.Tensor,        # [N, 19264] float32 (AIDO.Cell tokenized)
        pert_positions: torch.Tensor,   # [N] int64 (-1 if gene not in AIDO vocab)
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.string_embs = string_embs
        self.string_valid = string_valid
        self.input_ids = input_ids
        self.pert_positions = pert_positions
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            arr = np.array(raw_labels, dtype=np.int8) + 1   # {-1,0,1} → {0,1,2}
            self.labels = torch.from_numpy(arr).long()       # [N, 6640]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "string_emb": self.string_embs[idx],    # [256] float32
            "string_valid": self.string_valid[idx],  # bool scalar
            "input_ids": self.input_ids[idx],        # [19264] float32
            "pert_pos": self.pert_positions[idx],    # int64 (-1 if unknown)
        }
        if not self.is_test:
            item["label"] = self.labels[idx]         # [6640] int64
        return item


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        # Resolve data_dir relative to the project root (script's parent's parent).
        # This handles both cases:
        #   1. When data/ is a sibling of the mcts/ directory (MCTS working_node layout)
        #   2. When data/ is next to the script (monolithic layout)
        _script_dir = Path(__file__).resolve().parent
        _project_root = _script_dir.parent.parent  # up from mcts/nodeN/ to project root
        _cwd_data = Path.cwd() / data_dir
        if _cwd_data.exists():
            self.data_dir = _cwd_data
        elif (_project_root / data_dir).exists():
            self.data_dir = _project_root / data_dir
        else:
            self.data_dir = Path(data_dir)  # fall back to provided value
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

        # STRING_GNN state (set in setup)
        self._string_emb_matrix: Optional[torch.Tensor] = None
        self._string_node_to_idx: Optional[Dict[str, int]] = None

    def _init_string_gnn(self) -> None:
        """Extract STRING_GNN embeddings with rank-0-first barrier synchronization."""
        if self._string_emb_matrix is not None:
            return

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            print("Extracting STRING_GNN embeddings (once)...")
            emb, node_to_idx = extract_string_gnn_embeddings()
            self._string_emb_matrix = emb
            self._string_node_to_idx = node_to_idx
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if local_rank != 0:
            emb, node_to_idx = extract_string_gnn_embeddings()
            self._string_emb_matrix = emb
            self._string_node_to_idx = node_to_idx

    def _init_tokenizer(self) -> AutoTokenizer:
        """Rank-0-first tokenizer initialization."""
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)

    def _get_string_data(self, pert_ids: List[str]) -> tuple:
        """
        Look up STRING_GNN raw embeddings for each pert_id.
        Unknown genes get zero vector (model replaces with learnable null_raw in forward).
        """
        zero_emb = torch.zeros(STRING_DIM, dtype=torch.float32)
        embs = []
        valid_flags = []
        for pid in pert_ids:
            if pid in self._string_node_to_idx:
                idx = self._string_node_to_idx[pid]
                embs.append(self._string_emb_matrix[idx])
                valid_flags.append(True)
            else:
                embs.append(zero_emb)
                valid_flags.append(False)

        n_found = sum(valid_flags)
        print(f"  STRING vocab coverage: {n_found}/{len(pert_ids)} ({100.0*n_found/len(pert_ids):.1f}%)")
        return torch.stack(embs, dim=0), torch.tensor(valid_flags, dtype=torch.bool)

    def _tokenize_and_get_positions(
        self,
        tokenizer: AutoTokenizer,
        pert_ids: List[str],
        split_name: str = "split",
    ) -> tuple:
        """Tokenize perturbation IDs and find each gene's positional slot in AIDO vocab."""
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        chunk_size = 128
        all_input_ids: List[torch.Tensor] = []
        for i in range(0, len(expr_dicts), chunk_size):
            chunk = expr_dicts[i:i + chunk_size]
            toks = tokenizer(chunk, return_tensors="pt")
            all_input_ids.append(toks["input_ids"])  # [chunk, 19264] float32

        input_ids = torch.cat(all_input_ids, dim=0)  # [N, 19264] float32
        # Find position of perturbed gene: only slot where input_ids > -0.5
        non_missing = input_ids > -0.5
        has_gene = non_missing.any(dim=1)
        pert_positions = non_missing.long().argmax(dim=1)  # [N] int64
        pert_positions[~has_gene] = -1  # fallback for genes outside AIDO vocab

        coverage = 100.0 * has_gene.float().mean().item()
        print(f"  [{split_name}] AIDO vocab coverage: "
              f"{has_gene.sum().item()}/{len(pert_ids)} ({coverage:.1f}%)")
        return input_ids, pert_positions

    def setup(self, stage: Optional[str] = None) -> None:
        self._init_string_gnn()
        tokenizer = self._init_tokenizer()

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df   = pd.read_csv(self.data_dir / "val.tsv",   sep="\t")

            print("Preparing train set...")
            tr_str_embs, tr_str_valid = self._get_string_data(train_df["pert_id"].tolist())
            tr_ids, tr_pos = self._tokenize_and_get_positions(
                tokenizer, train_df["pert_id"].tolist(), "train")

            print("Preparing val set...")
            va_str_embs, va_str_valid = self._get_string_data(val_df["pert_id"].tolist())
            va_ids, va_pos = self._tokenize_and_get_positions(
                tokenizer, val_df["pert_id"].tolist(), "val")

            self.train_ds = PerturbationDataset(
                train_df, tr_str_embs, tr_str_valid, tr_ids, tr_pos, is_test=False)
            self.val_ds = PerturbationDataset(
                val_df, va_str_embs, va_str_valid, va_ids, va_pos, is_test=False)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            print("Preparing test set...")
            te_str_embs, te_str_valid = self._get_string_data(test_df["pert_id"].tolist())
            te_ids, te_pos = self._tokenize_and_get_positions(
                tokenizer, test_df["pert_id"].tolist(), "test")

            self.test_ds = PerturbationDataset(
                test_df, te_str_embs, te_str_valid, te_ids, te_pos, is_test=True)
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols  = test_df["symbol"].tolist()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.micro_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
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


# ─────────────────────────────────────────────────────────────────────────────
# Model: AIDO.Cell-10M Minimal LoRA + STRING_GNN Adapter + Compact Head
# ─────────────────────────────────────────────────────────────────────────────
class FusionDEGModel(nn.Module):
    """
    Dual-encoder DEG predictor:

      AIDO.Cell-10M (LoRA r=4, last 2 layers) → dual pool → [B, 512]
      STRING_GNN (frozen) → raw emb lookup → [B, 256]
                            learnable null_raw for unknowns
                            → linear adapter → [B, 128]
      cat([512, 128]) → [B, 640]
      → LayerNorm → Linear(32) → GELU → Dropout(0.6) → Linear(19920) → [B, 3, 6640]

    Trainable parameters:
      LoRA (r=4, Q/K/V, layers 6-7): ~12K
      STRING adapter (256→128):      ~33K
      string_null_raw:               256
      Head (LayerNorm + 2 Linears):  ~679K
      Total:                         ~724K (~483 params/training sample)
    """

    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        lora_layers: Optional[List[int]] = None,
        head_rank: int = 32,
        head_dropout: float = 0.6,
    ):
        super().__init__()
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # Default: last 2 layers of AIDO.Cell-10M (which has 8 layers indexed 0-7)
        self.lora_layers = lora_layers if lora_layers is not None else [6, 7]
        self.head_rank = head_rank
        self.head_dropout = head_dropout

        # ── STRING_GNN adapter (learnable) ────────────────────────────────────
        # Learns which PPI feature dimensions are predictive for DEG responses
        self.string_adapter = nn.Linear(STRING_DIM, ADAPTED_STRING_DIM)
        # Learnable null embedding for genes not in STRING vocab (~6-8% of genes)
        self.string_null_raw = nn.Parameter(torch.zeros(STRING_DIM))

        # ── Low-rank output head ──────────────────────────────────────────────
        # Input: [B, FUSION_DIM=640]; Output: [B, N_CLASSES * N_GENES]
        # Low-rank factorization: 640 → 32 → 19920 (~679K params vs 12.7M direct)
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Linear(FUSION_DIM, head_rank),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_rank, N_CLASSES * N_GENES),
        )

        # Truncated-normal init for stable early training
        nn.init.trunc_normal_(self.string_adapter.weight, std=0.02)
        if self.string_adapter.bias is not None:
            nn.init.zeros_(self.string_adapter.bias)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Backbone initialized separately (in LightningModule.setup)
        self.backbone: Optional[nn.Module] = None

    def initialize_backbone(self) -> None:
        """Load AIDO.Cell-10M with ultra-minimal LoRA (r=4, last 2 of 8 layers)."""
        from peft import LoraConfig, get_peft_model, TaskType

        backbone = AutoModel.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False

        # Monkey-patch get_input_embeddings for PEFT compatibility.
        # AIDO.Cell uses a custom GeneEmbedding, not a standard word embedding table,
        # so PEFT's enable_input_require_grads() would fail without this patch.
        _gene_emb = backbone.bert.gene_embedding
        backbone.get_input_embeddings = lambda: _gene_emb

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["query", "key", "value"],
            # layers_to_transform=[6, 7] = last 2 of 8 layers
            # flash_self shares weight tensors with self, so LoRA applies to both
            layers_to_transform=self.lora_layers,
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Cast LoRA adapter weights to float32 for stable optimization
        # (backbone stays in bfloat16; only LoRA deltas are float32)
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Gradient checkpointing: reduces activation memory at ~30% speed cost
        # Safe on AIDO.Cell-10M (small model, GC overhead minimal)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, 19264] float32
        pert_positions: torch.Tensor,  # [B] int64 (-1 for unknown genes)
        string_emb: torch.Tensor,      # [B, 256] float32 (zeros for unknown genes)
        string_valid: torch.Tensor,    # [B] bool
    ) -> torch.Tensor:
        """Returns logits: [B, 3, N_GENES]."""
        B = input_ids.size(0)
        device = input_ids.device

        # ── AIDO.Cell backbone ────────────────────────────────────────────────
        # Construct all-ones attention mask (AIDO overrides it internally anyway)
        attention_mask = torch.ones(B, input_ids.size(1), dtype=torch.long, device=device)
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, 19266, 256] bfloat16

        # ── Dual pooling ─────────────────────────────────────────────────────
        # Global mean-pool over 19264 gene positions (exclude 2 appended summary tokens)
        mean_pool = hidden[:, :19264, :].mean(dim=1).float()  # [B, 256]

        # Per-gene positional extraction
        valid_pos = pert_positions >= 0                       # [B] bool
        safe_pos = pert_positions.clamp(min=0)                # avoid -1 index
        gene_emb_raw = hidden[
            torch.arange(B, device=device), safe_pos, :
        ].float()  # [B, 256]

        # Differentiable masking: unknown genes fall back to mean_pool
        valid_f = valid_pos.float().unsqueeze(-1)             # [B, 1]
        gene_emb = gene_emb_raw * valid_f + mean_pool * (1.0 - valid_f)  # [B, 256]

        aido_dual = torch.cat([gene_emb, mean_pool], dim=-1)  # [B, 512]

        # ── STRING_GNN adapter ────────────────────────────────────────────────
        # For unknown STRING genes, substitute learnable null_raw (trainable)
        # string_emb contains zeros for unknown genes; masked out here
        string_valid_f = string_valid.float().unsqueeze(-1).to(device)  # [B, 1]
        raw_string = (
            string_emb.to(device) * string_valid_f
            + self.string_null_raw.unsqueeze(0) * (1.0 - string_valid_f)
        )  # [B, 256] — differentiable blend
        adapted_string = self.string_adapter(raw_string)  # [B, 128]

        # ── Fusion + head ─────────────────────────────────────────────────────
        combined = torch.cat([aido_dual, adapted_string], dim=-1)  # [B, 640]
        logits = self.head(combined)                               # [B, 3*6640]
        return logits.view(B, N_CLASSES, N_GENES)                  # [B, 3, 6640]

    def get_parameter_groups(
        self, lr_backbone: float, lr_head: float, weight_decay: float
    ) -> List[Dict]:
        """
        Two optimizer groups:
          Group 0 (LoRA backbone): very low LR to gently adapt pretrained features
          Group 1 (STRING adapter + head): standard LR for randomly-initialized params
        """
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_and_adapter_params = (
            list(self.head.parameters())
            + list(self.string_adapter.parameters())
            + [self.string_null_raw]
        )
        return [
            {"params": backbone_params,       "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_and_adapter_params, "lr": lr_head,  "weight_decay": weight_decay},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        lora_layers: Optional[List[int]] = None,
        head_rank: int = 32,
        head_dropout: float = 0.6,
        lr_backbone: float = 5e-5,
        lr_head: float = 5e-4,
        weight_decay: float = 5e-2,
        gamma_focal: float = 3.0,
        label_smoothing: float = 0.10,
        max_epochs: int = 120,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model: Optional[FusionDEGModel] = None
        self.criterion: Optional[FocalLoss] = None

        # Accumulators (cleared each epoch)
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = FusionDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                lora_layers=self.hparams.lora_layers,
                head_rank=self.hparams.head_rank,
                head_dropout=self.hparams.head_dropout,
            )
            self.model.initialize_backbone()
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        if stage == "test" and hasattr(self.trainer.datamodule, "test_pert_ids"):
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols

    def forward(
        self,
        input_ids: torch.Tensor,
        pert_positions: torch.Tensor,
        string_emb: torch.Tensor,
        string_valid: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, pert_positions, string_emb, string_valid)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits: [B, 3, G]; labels: [B, G] ({0,1,2}) → scalar loss."""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
        labels_flat = labels.reshape(-1)                        # [B*G]
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["input_ids"], batch["pert_pos"],
            batch["string_emb"], batch["string_valid"],
        )
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["pert_pos"],
            batch["string_emb"], batch["string_valid"],
        )
        loss = self._compute_loss(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, G]
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds  = torch.cat(self._val_preds,   dim=0)  # [N_local, 3, G]
        local_labels = torch.cat(self._val_labels,  dim=0)  # [N_local, G]
        local_idx    = torch.cat(self._val_indices, dim=0)  # [N_local]

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        world_size = self.trainer.world_size if self.trainer.world_size else 1
        if world_size > 1:
            all_preds  = self.all_gather(local_preds)   # [world, N_local, 3, G]
            all_labels = self.all_gather(local_labels)   # [world, N_local, G]
            all_idx    = self.all_gather(local_idx)      # [world, N_local]

            preds_flat  = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
            labels_flat = all_labels.view(-1, N_GENES).cpu().numpy()
            idx_flat    = all_idx.view(-1).cpu().numpy()

            # De-duplicate and restore original order
            unique_pos  = np.unique(idx_flat, return_index=True)[1]
            preds_flat  = preds_flat[unique_pos]
            labels_flat = labels_flat[unique_pos]
            order       = np.argsort(idx_flat[unique_pos])
            preds_flat  = preds_flat[order]
            labels_flat = labels_flat[order]

            f1 = compute_deg_f1(preds_flat, labels_flat)
            self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        else:
            f1 = compute_deg_f1(local_preds.numpy(), local_labels.numpy())
            self.log("val_f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["pert_pos"],
            batch["string_emb"], batch["string_valid"],
        )
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, G]
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds,   dim=0)
        local_idx   = torch.cat(self._test_indices, dim=0)

        all_preds = self.all_gather(local_preds)
        all_idx   = self.all_gather(local_idx)

        self._test_preds.clear()
        self._test_indices.clear()

        if self.trainer.is_global_zero:
            preds = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
            idxs  = all_idx.view(-1).cpu().numpy()

            # De-duplicate (DDP may overlap last batch) and restore order
            unique_pos  = np.unique(idxs, return_index=True)[1]
            preds       = preds[unique_pos]
            sorted_idxs = idxs[unique_pos]
            order       = np.argsort(sorted_idxs)
            preds       = preds[order]
            final_idxs  = sorted_idxs[order]

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "test_predictions.tsv"

            rows = []
            for rank_i, orig_i in enumerate(final_idxs):
                rows.append({
                    "idx":        self._test_pert_ids[orig_i],
                    "input":      self._test_symbols[orig_i],
                    "prediction": json.dumps(preds[rank_i].tolist()),
                })
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {out_path}")

    def configure_optimizers(self):
        param_groups = self.model.get_parameter_groups(
            lr_backbone=self.hparams.lr_backbone,
            lr_head=self.hparams.lr_head,
            weight_decay=self.hparams.weight_decay,
        )
        opt = torch.optim.AdamW(param_groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    # ── Checkpoint: save only trainable parameters ────────────────────────────
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters + persistent buffers (skip frozen backbone)."""
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                k = prefix + name
                if k in full:
                    trainable[k] = full[k]
        for name, buf in self.named_buffers():
            k = prefix + name
            if k in full:
                trainable[k] = full[k]
        total   = sum(p.numel() for p in self.parameters())
        tr_cnt  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Checkpoint: saving {tr_cnt:,}/{total:,} params "
            f"({100.0 * tr_cnt / max(total, 1):.2f}% trainable)"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        """Load only trainable parameters; frozen backbone is re-initialized from pretrained."""
        return super().load_state_dict(state_dict, strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AIDO.Cell-10M Minimal LoRA + STRING_GNN Adapter DEG predictor"
    )
    p.add_argument("--data-dir",               type=str,   default="data")
    p.add_argument("--micro-batch-size",        type=int,   default=16)
    p.add_argument("--global-batch-size",       type=int,   default=128)
    p.add_argument("--max-epochs",              type=int,   default=120)
    p.add_argument("--lr-backbone",             type=float, default=5e-5)
    p.add_argument("--lr-head",                 type=float, default=5e-4)
    p.add_argument("--weight-decay",            type=float, default=5e-2)
    p.add_argument("--lora-r",                  type=int,   default=4)
    p.add_argument("--lora-alpha",              type=int,   default=8)
    p.add_argument("--lora-dropout",            type=float, default=0.05)
    p.add_argument("--head-rank",               type=int,   default=32)
    p.add_argument("--head-dropout",            type=float, default=0.6)
    p.add_argument("--gamma-focal",             type=float, default=3.0)
    p.add_argument("--label-smoothing",         type=float, default=0.10)
    p.add_argument("--early-stopping-patience", type=int,   default=20)
    p.add_argument("--num-workers",             type=int,   default=4)
    p.add_argument("--val-check-interval",      type=float, default=1.0)
    p.add_argument("--debug-max-step",          type=int,   default=None)
    p.add_argument("--fast-dev-run",            action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    pl.seed_everything(0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Distributed setup ─────────────────────────────────────────────────────
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps    = args.debug_max_step if args.debug_max_step is not None else -1
    limit_train = limit_val = limit_test = 1.0
    if args.debug_max_step is not None:
        limit_train = args.debug_max_step
        limit_val   = args.debug_max_step
        limit_test  = args.debug_max_step

    val_check_interval = args.val_check_interval if (
        args.debug_max_step is None and not args.fast_dev_run
    ) else 1.0

    # ── Strategy ──────────────────────────────────────────────────────────────
    if n_gpus == 1:
        strategy = SingleDeviceStrategy(device="cuda:0")
    else:
        strategy = DDPStrategy(
            find_unused_parameters=True,    # LoRA leaves some backbone params unused
            timeout=timedelta(seconds=300),
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="fusion-deg-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stopping_patience,
        verbose=True,
    )
    lr_monitor   = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # ── Loggers ───────────────────────────────────────────────────────────────
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    # ── Data & model ──────────────────────────────────────────────────────────
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=[6, 7],  # last 2 of 8 layers in AIDO.Cell-10M
        head_rank=args.head_rank,
        head_dropout=args.head_dropout,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── Test ───────────────────────────────────────────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # ── Save test score ────────────────────────────────────────────────────────
    if test_results and trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        with open(score_path, "w") as f:
            f.write(f"test_results: {test_results}\n")
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
