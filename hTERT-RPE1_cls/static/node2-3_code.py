"""
Node 2-3 – AIDO.Cell-100M + LoRA r=32 + Enriched Representation
           (Gene-Pos + Mean-Pool Concat) + Bilinear Head (dim=512)
           + Focal Loss + Mild Class Weights

Key improvements over parent (node2, F1=0.3445):
  1. Multi-gene input (all=1.0, perturbed=10.0) – fixes OOD single-gene probe
  2. LoRA r=32 (vs r=16) – richer backbone adaptation
  3. Enriched 1280-dim representation:
       concat(gene-position embedding [640], mean-pool over all 19264 gene positions [640])
     - gene-pos: perturbed gene's specific contextual embedding (validated as best in node2-1)
     - mean-pool: global co-expression summary (adds missing global context)
  4. Bilinear head (dim=512): ~6M params vs 2.4M (node2-2) and 42M (node2-1)
     Provides structural inductive bias without head over-parameterization
  5. Focal loss (γ=2.0) + label smoothing (0.05) + mild class weights [2.0, 1.0, 4.0]
  6. Cosine LR schedule (T_max≈100 epochs) with linear warmup (100 steps)
  7. Gradient clipping (max_norm=1.0) for training stability

Differentiation from siblings:
  vs. node2-1 (F1=0.4234, wide MLP 42M params → val loss diverges at epoch 3):
    - Bilinear head (~6M params) instead of wide MLP
    - Enriched representation: adds mean-pool concatenation
    - Mild class weights added to focal loss
  vs. node2-2 (F1=0.4102, bilinear dim=256 → capacity ceiling at epoch 22):
    - Bilinear dim=512 (vs 256) – directly addresses diagnosed capacity bottleneck
    - Enriched representation: adds global mean-pool (NEW in this lineage)
    - Mild class weights added (node2-2's Priority 4 recommendation)
    - Cosine T_max=100 epochs (vs 60 in node2-2)
    - patience=30 (vs 25)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import math
import argparse
from pathlib import Path
from datetime import timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ─── Constants ────────────────────────────────────────────────────────────────

AIDO_CELL_DIR     = "/home/Models/AIDO.Cell-100M"
N_GENES_OUT       = 6640
N_CLASSES         = 3
HIDDEN_SIZE       = 640    # AIDO.Cell-100M hidden dimension
N_LAYERS          = 18     # total transformer layers
FUSION_LAYERS     = 6      # trailing layers to fuse
REPR_DIM          = HIDDEN_SIZE * 2  # 1280: gene-pos (640) + mean-pool (640)

LORA_R            = 32
LORA_ALPHA        = 64
LORA_DROPOUT      = 0.05

HEAD_HIDDEN_DIM   = 768
BILINEAR_DIM      = 512    # 2× node2-2's 256 → addresses capacity bottleneck
HEAD_DROPOUT      = 0.2

FOCAL_GAMMA       = 2.0
LABEL_SMOOTHING   = 0.05

# Mild class weights [down-reg(−1→0), neutral(0→1), up-reg(+1→2)]
# Distributions: neutral=88.9%, down=8.14%, up=3.0%
# Mild upweighting as recommended by node2-2: [~2×, 1×, ~4×]
CLASS_WEIGHTS     = torch.tensor([2.0, 1.0, 4.0], dtype=torch.float32)

PERTURB_EXPR      = 10.0   # expression level for perturbed gene
BASELINE_EXPR     = 1.0    # expression level for all other genes


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Per-gene macro F1 matching calc_metric.py.  pred_np: [N,3,G], labels_np: [N,G]."""
    pred_cls = pred_np.argmax(axis=1)
    f1_vals = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]
        yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Focal CE Loss ────────────────────────────────────────────────────────────

def focal_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weight: Optional[torch.Tensor],
    gamma: float,
    label_smoothing: float,
) -> torch.Tensor:
    """
    Focal cross-entropy loss with optional class weights and label smoothing.
    logits: [N, C]  targets: [N]
    """
    # Standard CE with label smoothing and per-class weight
    ce = F.cross_entropy(
        logits, targets,
        weight=class_weight,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    # Focal weight: (1 - p_t)^gamma, where p_t = probability of the true class
    with torch.no_grad():
        pt = F.softmax(logits.float(), dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_w = (1.0 - pt) ** gamma
    return (focal_w * ce).mean()


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbDataset(Dataset):
    """Stores pre-built multi-gene AIDO.Cell inputs + gene-position indices."""

    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        input_ids: torch.Tensor,       # [N, 19264] float32
        gene_positions: torch.Tensor,  # [N] long
        labels: Optional[torch.Tensor] = None,  # [N, 6640] long
    ):
        self.pert_ids      = pert_ids
        self.symbols       = symbols
        self.input_ids     = input_ids
        self.gene_positions = gene_positions
        self.labels        = labels

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":       self.pert_ids[idx],
            "symbol":        self.symbols[idx],
            "input_ids":     self.input_ids[idx],
            "gene_position": self.gene_positions[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    pert_ids  = [b["pert_id"]        for b in batch]
    symbols   = [b["symbol"]         for b in batch]
    input_ids = torch.stack([b["input_ids"]     for b in batch])  # [B, 19264]
    gene_pos  = torch.stack([b["gene_position"] for b in batch])  # [B]
    out = {"pert_id": pert_ids, "symbol": symbols,
           "input_ids": input_ids, "gene_position": gene_pos}
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 4,
        num_workers: int = 2,
    ):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage: Optional[str] = None):
        # ── DDP-safe tokenizer loading ────────────────────────────────────
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)

        # ── Build gene → vocabulary index mapping ─────────────────────────
        # tokenizer.gene_to_index maps gene symbols to their 0-indexed position
        # in the 19264-dim AIDO.Cell vocabulary.
        gene_to_idx = getattr(tokenizer, "gene_to_index", {})

        def build_inputs(symbols: List[str]):
            """
            Build multi-gene AIDO.Cell inputs for a list of perturbed gene symbols.
            Strategy: all 19,264 genes at expression=1.0 (biologically plausible baseline);
                      perturbed gene elevated to 10.0 (clear perturbation signal).
            After CP10K normalization inside AIDO.Cell model:
              baseline → log1p(1.0/19273*10000) ≈ 0.417
              perturbed → log1p(10.0/19273*10000) ≈ 1.877
            Returns: input_ids [N, 19264] float32, gene_positions [N] long
            """
            n = len(symbols)
            # All genes at baseline expression
            ids = torch.full((n, 19264), BASELINE_EXPR, dtype=torch.float32)
            gene_positions = torch.zeros(n, dtype=torch.long)

            for i, sym in enumerate(symbols):
                if sym in gene_to_idx:
                    pos = gene_to_idx[sym]
                else:
                    # OOV fallback: use single-gene tokenizer to find vocabulary position
                    tok_out = tokenizer(
                        {"gene_names": [sym], "expression": [1.0]}, return_tensors="pt"
                    )
                    single_ids = tok_out["input_ids"][0]  # [19264] float32
                    # Position where expression > 0.5 (only the expressed gene)
                    pos = int((single_ids > 0.5).float().argmax().item())

                ids[i, pos] = PERTURB_EXPR
                gene_positions[i] = pos

            return ids, gene_positions

        # ── Load splits ───────────────────────────────────────────────────
        def load_split(fname: str, has_label: bool) -> PerturbDataset:
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            symbols  = df["symbol"].tolist()
            pert_ids = df["pert_id"].tolist()
            ids, gpos = build_inputs(symbols)
            labels = None
            if has_label and "label" in df.columns:
                rows = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return PerturbDataset(pert_ids, symbols, ids, gpos, labels)

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Bilinear Prediction Head ─────────────────────────────────────────────────

class BilinearPerturbHead(nn.Module):
    """
    Bilinear interaction head maps enriched perturbation representation to logits.

    Input: [B, input_dim=1280]
    Architecture:
        LayerNorm(1280)
        → Linear(1280, head_hidden=768) → GELU → Dropout(0.2)
        → Linear(768, n_classes×bilinear_dim=1536)
        → LayerNorm(1536) → reshape [B, 3, 512]
        × Embedding(6640, 512)^T
        → logits [B, 3, 6640]

    Inductive bias: response of gene g = f(perturbation_embedding, gene_g_identity),
    which generalizes better with 1,416 training samples than a dense MLP.

    Head parameter count:
        LN(1280): 2,560
        Linear(1280→768): 983,808
        Linear(768→1536): 1,181,184
        LN(1536): 3,072
        Embedding(6640, 512): 3,399,680
        Total: ~5.6M  (vs 42M in node2-1, vs 2.4M in node2-2)
    """

    def __init__(
        self,
        input_dim: int   = REPR_DIM,
        head_hidden: int = HEAD_HIDDEN_DIM,
        bilinear_dim: int = BILINEAR_DIM,
        n_genes_out: int  = N_GENES_OUT,
        n_classes: int    = N_CLASSES,
        dropout: float    = HEAD_DROPOUT,
    ):
        super().__init__()
        self.n_classes    = n_classes
        self.bilinear_dim = bilinear_dim

        self.norm_in  = nn.LayerNorm(input_dim)
        self.proj1    = nn.Linear(input_dim, head_hidden)
        self.act      = nn.GELU()
        self.drop     = nn.Dropout(dropout)
        self.proj2    = nn.Linear(head_hidden, n_classes * bilinear_dim)
        self.norm_out = nn.LayerNorm(n_classes * bilinear_dim)

        # Learnable output gene embeddings (one per output gene position)
        self.out_gene_emb = nn.Embedding(n_genes_out, bilinear_dim)
        # Initialize with small uniform values for stable start
        nn.init.uniform_(self.out_gene_emb.weight, -0.05, 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, input_dim] → logits [B, n_classes, n_genes_out]"""
        x = self.norm_in(x)                                        # [B, 1280]
        x = self.proj1(x)                                          # [B, 768]
        x = self.act(x)
        x = self.drop(x)
        x = self.proj2(x)                                          # [B, n_classes*512]
        x = self.norm_out(x)
        x = x.view(x.shape[0], self.n_classes, self.bilinear_dim)  # [B, 3, 512]

        gene_emb = self.out_gene_emb.weight                        # [6640, 512]
        logits = torch.einsum("bcd,gd->bcg", x, gene_emb)          # [B, 3, 6640]
        return logits


# ─── Model ────────────────────────────────────────────────────────────────────

class AIDOCellEnrichedModel(nn.Module):
    """
    AIDO.Cell-100M + LoRA r=32 with multi-layer fusion and enriched representation.

    Key architectural features:
      1. Multi-gene input: all 19,264 genes at 1.0; perturbed gene at 10.0
         (constructed in DataModule — model receives pre-built float32 input_ids)
      2. LoRA r=32 on Q/K/V across all 18 layers (~2.21M trainable backbone params)
      3. Weighted fusion of last 6 transformer layers (learnable softmax weights)
      4. Enriched representation:
           concat(gene-pos[B,640], mean-pool[B,640]) → [B,1280]
         - gene-pos: hidden state at perturbed gene's vocabulary index (discriminative)
         - mean-pool: average over all 19264 gene positions (global co-expression context)
      5. Bilinear prediction head (dim=512) → logits [B,3,6640]
    """

    def __init__(self):
        super().__init__()
        # ── Load and configure backbone ───────────────────────────────────
        backbone = AutoModel.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False

        # Patch enable_input_require_grads for PEFT compatibility
        # AIDO.Cell does not implement get_input_embeddings(); this patch registers
        # a forward hook on the gene embedding layer to ensure gradient flow.
        def _safe_enable_input_require_grads():
            def _make_inputs_require_grad(module, inp, out):
                if isinstance(out, torch.Tensor):
                    out.requires_grad_(True)
            backbone.bert.gene_embedding.register_forward_hook(_make_inputs_require_grad)

        backbone.enable_input_require_grads = _safe_enable_input_require_grads

        # ── Apply LoRA r=32 on Q/K/V across all 18 layers ─────────────────
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["query", "key", "value"],
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Enable gradient checkpointing for memory efficiency
        self.backbone.base_model.model.config.use_cache = False
        self.backbone.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA params to float32 for optimizer stability
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Learnable layer-fusion weights ────────────────────────────────
        self.layer_weights = nn.Parameter(torch.zeros(FUSION_LAYERS))

        # ── Bilinear prediction head ──────────────────────────────────────
        # Input: enriched 1280-dim representation (gene-pos 640 + mean-pool 640)
        self.head = BilinearPerturbHead(
            input_dim=REPR_DIM,         # 1280
            head_hidden=HEAD_HIDDEN_DIM,  # 768
            bilinear_dim=BILINEAR_DIM,  # 512
            n_genes_out=N_GENES_OUT,    # 6640
            n_classes=N_CLASSES,        # 3
            dropout=HEAD_DROPOUT,       # 0.2
        )

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, 19264] float32
        gene_positions: torch.Tensor,  # [B] long
    ) -> torch.Tensor:
        """Returns logits [B, 3, 6640]."""
        B = input_ids.shape[0]
        attn_mask = torch.ones(B, input_ids.shape[1], dtype=torch.long,
                               device=input_ids.device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )

        # hidden_states: tuple of 19 tensors [B, 19266, 640]
        # indices 0..18; use last FUSION_LAYERS = indices [13..18]
        hidden_states = torch.stack(
            [out.hidden_states[i].float()
             for i in range(N_LAYERS - FUSION_LAYERS + 1, N_LAYERS + 1)],
            dim=0,
        )  # [FUSION_LAYERS, B, 19266, 640]

        # Weighted combination across fusion layers
        weights = torch.softmax(self.layer_weights, dim=0)           # [FUSION_LAYERS]
        fused = (hidden_states * weights[:, None, None, None]).sum(0)  # [B, 19266, 640]

        # ── Enriched representation ────────────────────────────────────────
        # 1. Gene-position: extract hidden state at perturbed gene's vocabulary index
        #    This captures the perturbation-specific contextual embedding
        gene_pos_repr = fused[torch.arange(B, device=fused.device), gene_positions, :]
        # gene_pos_repr: [B, 640]

        # 2. Mean-pool: average over all 19264 gene positions (exclude 2 summary tokens)
        #    This provides the global co-expression context
        gene_mean_repr = fused[:, :19264, :].mean(dim=1)
        # gene_mean_repr: [B, 640]

        # 3. Concatenate → 1280-dim enriched representation
        gene_repr = torch.cat([gene_pos_repr, gene_mean_repr], dim=-1)
        # gene_repr: [B, 1280]

        # ── Bilinear head → logits ─────────────────────────────────────────
        return self.head(gene_repr)  # [B, 3, 6640]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
    """Gather and un-pad tensors from all DDP ranks."""
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))

    pad = max_sz - local_p.shape[0]
    p = local_p.to(device)
    l = local_l.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], 0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], 0)

    gp = [torch.zeros_like(p) for _ in range(world_size)]
    gl = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(gp, p)
    dist.all_gather(gl, l)

    rp = torch.cat([gp[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    rl = torch.cat([gl[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    return rp, rl


# ─── LightningModule ──────────────────────────────────────────────────────────

class AIDOCellLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_backbone: float = 5e-5,
        lr_head: float = 3e-4,
        weight_decay: float = 1e-3,
        focal_gamma: float = FOCAL_GAMMA,
        label_smoothing: float = LABEL_SMOOTHING,
        warmup_steps: int = 100,
        max_steps_cosine: int = 4400,  # T_max for cosine schedule (≈100 epochs)
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []
        self._test_labels:   List[torch.Tensor] = []

    def setup(self, stage: Optional[str] = None):
        self.model = AIDOCellEnrichedModel()
        # Register class weights as buffer for automatic device placement
        self.register_buffer("class_weights", CLASS_WEIGHTS)

    def forward(self, input_ids, gene_positions):
        return self.model(input_ids, gene_positions)

    def _loss(self, logits, labels):
        """Focal CE loss with mild class weights and label smoothing."""
        # Reshape to 2D for deterministic nll_loss CUDA kernel
        # logits: [B, 3, 6640] → [B*6640, 3];  labels: [B, 6640] → [B*6640]
        logits_2d = logits.float().permute(0, 2, 1).reshape(-1, N_CLASSES)
        labels_1d = labels.reshape(-1)
        return focal_ce_loss(
            logits_2d, labels_1d,
            class_weight=self.class_weights.to(logits_2d.device),
            gamma=self.hparams.focal_gamma,
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"])
        if "label" in batch:
            loss = self._loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        lp = torch.cat(self._val_preds,  0)
        ll = torch.cat(self._val_labels, 0)
        if self.trainer.world_size > 1:
            lp, ll = _gather_tensors(lp, ll, self.device, self.trainer.world_size)
        f1 = compute_per_gene_f1(lp.numpy(), ll.numpy())
        # sync_dist=True: all ranks already hold the global result from all_gather
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"])
        probs  = torch.softmax(logits, dim=1)
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs  = torch.cat(self._test_preds, 0)
        dummy_labels = (torch.cat(self._test_labels, 0) if self._test_labels
                        else torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long))

        if self.trainer.world_size > 1:
            all_probs, all_labels = _gather_tensors(
                local_probs, dummy_labels, self.device, self.trainer.world_size
            )
            all_pert = [None] * self.trainer.world_size
            all_syms = [None] * self.trainer.world_size
            dist.all_gather_object(all_pert, self._test_pert_ids)
            dist.all_gather_object(all_syms, self._test_symbols)
            all_pert = [p for sub in all_pert for p in sub]
            all_syms = [s for sub in all_syms for s in sub]
        else:
            all_probs, all_labels = local_probs, dummy_labels
            all_pert, all_syms   = self._test_pert_ids, self._test_symbols

        if self.trainer.is_global_zero:
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate by pert_id (DDP sampler may pad with duplicates)
            seen_pids: set = set()
            dedup_indices: List[int] = []
            for i, pid in enumerate(all_pert):
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    dedup_indices.append(i)

            all_probs_np = all_probs.numpy()
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for i in dedup_indices:
                    fh.write(
                        f"{all_pert[i]}\t{all_syms[i]}\t"
                        f"{json.dumps(all_probs_np[i].tolist())}\n"
                    )
            self.print(f"[Node2-3] Saved {len(dedup_indices)} test predictions → {pred_path}")

            # Self-evaluate if test labels are available
            if all_labels.any():
                dedup_probs  = all_probs_np[dedup_indices]
                dedup_labels = all_labels[dedup_indices].numpy()
                f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
                self.print(f"[Node2-3] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # ── Parameter groups: backbone (LoRA) vs head+fusion ──────────────
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone"):
                backbone_params.append(param)
            else:
                # layer_weights and head.* → fresh parameters → head LR
                head_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": hp.lr_backbone},
                {"params": head_params,     "lr": hp.lr_head},
            ],
            weight_decay=hp.weight_decay,
        )

        # ── Cosine annealing with linear warmup ───────────────────────────
        # Calibrated to T_max ≈ 100 epochs to ensure meaningful LR decay
        # before early stopping triggers (avoids node2-1's issue where LR
        # was still at 92% of peak when training stopped)
        def lr_lambda(current_step: int) -> float:
            if current_step < hp.warmup_steps:
                return float(current_step) / float(max(1, hp.warmup_steps))
            progress = float(current_step - hp.warmup_steps) / float(
                max(1, hp.max_steps_cosine - hp.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and buffers."""
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        sd = {k: v for k, v in full_sd.items() if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving ckpt: {trained}/{total} params ({100*trained/total:.2f}%)")
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 2-3 – AIDO.Cell-100M LoRA + Enriched Repr + Bilinear Head"
    )
    p.add_argument("--data-dir",            type=str,   default="data")
    p.add_argument("--lr-backbone",         type=float, default=5e-5,
                   help="Learning rate for LoRA backbone parameters")
    p.add_argument("--lr-head",             type=float, default=3e-4,
                   help="Learning rate for head and fusion parameters")
    p.add_argument("--weight-decay",        type=float, default=1e-3)
    p.add_argument("--focal-gamma",         type=float, default=2.0)
    p.add_argument("--label-smoothing",     type=float, default=0.05)
    p.add_argument("--warmup-steps",        type=int,   default=100)
    p.add_argument("--micro-batch-size",    type=int,   default=4)
    p.add_argument("--global-batch-size",   type=int,   default=32,
                   help="Must be a multiple of micro_batch_size * 8")
    p.add_argument("--max-epochs",          type=int,   default=150)
    p.add_argument("--patience",            type=int,   default=30,
                   help="EarlyStopping patience (epochs)")
    p.add_argument("--max-steps-cosine",    type=int,   default=-1,
                   help="T_max for cosine LR; -1 = auto (steps_per_epoch * 100)")
    p.add_argument("--num-workers",         type=int,   default=2)
    p.add_argument("--val-check-interval",  type=float, default=1.0)
    p.add_argument("--debug-max-step",      type=int,   default=None,
                   help="Limit to this many steps (debug only)")
    p.add_argument("--fast-dev-run",        action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Compute cosine T_max ───────────────────────────────────────────────
    # Calibrate to ~100 epochs to ensure meaningful LR decay before early stopping.
    # With 1416 train samples / global_batch_size=32 ≈ 45 steps/epoch × 100 = 4500.
    if args.max_steps_cosine == -1:
        estimated_steps_per_epoch = math.ceil(1416 / args.global_batch_size)
        max_steps_cosine = estimated_steps_per_epoch * 100
    else:
        max_steps_cosine = args.max_steps_cosine

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # ── DataModule & LightningModule ─────────────────────────────────────
    dm = PerturbDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    lit = AIDOCellLitModule(
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        warmup_steps=args.warmup_steps,
        max_steps_cosine=max_steps_cosine,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max",
                           patience=args.patience, min_delta=1e-5)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    # ── Loggers ───────────────────────────────────────────────────────────
    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    # ── Debug / fast-dev-run settings ─────────────────────────────────────
    max_steps: int = -1
    limit_train: float | int = 1.0
    limit_val:   float | int = 1.0
    limit_test:  float | int = 1.0
    fast_dev_run = False

    if args.debug_max_step is not None:
        max_steps   = args.debug_max_step
        limit_train = args.debug_max_step
        limit_val   = 2
        limit_test  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    # ── Trainer ───────────────────────────────────────────────────────────
    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accum,
        gradient_clip_val=1.0,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    trainer.fit(lit, datamodule=dm)

    # ── Test ──────────────────────────────────────────────────────────────
    ckpt_path = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt_path)

    # ── Write stub test score ─────────────────────────────────────────────
    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 2-3 – AIDO.Cell-100M LoRA r=32 + Enriched Repr + Bilinear Head (dim=512)\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
