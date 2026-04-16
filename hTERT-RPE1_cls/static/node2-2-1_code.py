"""
Node 2-2-1 – AIDO.Cell-100M + LoRA r=64 + Enhanced Bilinear Head (bilinear_dim=512)

Improvements over parent (node2-2, F1=0.4102):
  1. Increased bilinear_dim 256 → 512: Each of the 3 class subspaces has 2x more capacity
     to distinguish perturbation effects across 6,640 output genes.
  2. Enriched perturbation representation: Concatenate gene-position embedding (640-dim)
     with learnable-attention-pooled mean over all 19,264 positions (640-dim), projected
     1280 → bilinear space. Provides both local (perturbation-specific) and global
     (co-expression context) signals, addressing the single gene-position extraction bottleneck.
  3. Mild class-weighted focal loss: weights [2.0, 0.5, 4.0] for classes down/neutral/up,
     targeting improvement in minority class (up-regulated, 3%) F1 contribution.
  4. Calibrated LR schedule to ~80 effective epochs (vs 60 in parent) to allow secondary
     improvement phase as LR decays into its lower range.
  5. Extended early stopping patience: 30 (vs 25 in parent).
  6. head_hidden_dim increased to 768 to accommodate larger bilinear projection.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import argparse
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional

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

AIDO_CELL_DIR = "/home/Models/AIDO.Cell-100M"
N_GENES_OUT   = 6640
N_CLASSES     = 3
HIDDEN_SIZE   = 640   # AIDO.Cell-100M hidden dim
N_LAYERS      = 18   # total transformer layers
FUSION_LAYERS = 6    # number of trailing layers to fuse

# Baseline expression values for realistic multi-gene input
BASELINE_EXPR  = 1.0   # all background genes
PERTURB_EXPR   = 10.0  # the perturbed gene (10× elevated above baseline)


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


# ─── Dataset ──────────────────────────────────────────────────────────────────

class AIDOCellPerturbDataset(Dataset):
    """Stores pert_ids, symbols, gene_vocab_positions, and labels for lazy batch construction."""

    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        gene_vocab_positions: List[int],  # index into AIDO.Cell 19264-gene vocabulary
        labels: Optional[torch.Tensor] = None,  # [N, 6640] long (class indices 0/1/2)
    ):
        self.pert_ids            = pert_ids
        self.symbols             = symbols
        self.gene_vocab_positions = gene_vocab_positions
        self.labels              = labels

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":            self.pert_ids[idx],
            "symbol":             self.symbols[idx],
            "gene_vocab_position": self.gene_vocab_positions[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def build_collate_fn(vocab_size: int = 19264):
    """
    Returns a collate_fn that constructs full 19264-gene expression profiles at batch time.
    All genes are set to BASELINE_EXPR; the perturbed gene is elevated to PERTURB_EXPR.
    This is far more memory-efficient than pre-storing [N, 19264] tensors in the dataset.
    """
    def collate_fn(batch):
        B = len(batch)
        pert_ids   = [b["pert_id"] for b in batch]
        symbols    = [b["symbol"]  for b in batch]
        gene_pos   = torch.tensor([b["gene_vocab_position"] for b in batch], dtype=torch.long)

        # Build full expression profile: all genes at baseline, perturbed gene at PERTURB_EXPR
        input_ids = torch.full((B, vocab_size), fill_value=BASELINE_EXPR, dtype=torch.float32)
        for i, pos in enumerate(gene_pos.tolist()):
            if 0 <= pos < vocab_size:
                input_ids[i, pos] = PERTURB_EXPR

        out = {
            "pert_id":    pert_ids,
            "symbol":     symbols,
            "input_ids":  input_ids,    # [B, 19264] float32
            "gene_position": gene_pos,  # [B] long – vocabulary index of the perturbed gene
        }
        if "label" in batch[0]:
            out["label"] = torch.stack([b["label"] for b in batch])  # [B, 6640] long
        return out
    return collate_fn


# ─── DataModule ───────────────────────────────────────────────────────────────

class AIDOCellDataModule(pl.LightningDataModule):

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
        # ── Load tokenizer (DDP-safe barrier) ─────────────────────────────────
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)

        # Build gene_to_index mapping from the tokenizer's vocabulary
        gene_to_index: Dict[str, int] = {}
        if hasattr(tokenizer, 'gene_to_index'):
            gene_to_index = tokenizer.gene_to_index  # symbol -> vocab position
        elif hasattr(tokenizer, 'vocab'):
            gene_to_index = tokenizer.vocab

        # ── Helper: find vocabulary position of a gene symbol ─────────────────
        def get_gene_pos(symbol: str) -> int:
            """Return the vocabulary index of a gene symbol; 0 as fallback for OOV."""
            if symbol in gene_to_index:
                return int(gene_to_index[symbol])
            # Fallback: tokenize the single-gene input and find where expression > 0
            single_input = [{"gene_names": [symbol], "expression": [PERTURB_EXPR]}]
            tok_out = tokenizer(single_input, return_tensors="pt")
            ids = tok_out["input_ids"]  # [1, 19264] float32
            pos = int((ids > (BASELINE_EXPR + 0.1)).float().argmax(dim=1).item())
            return pos

        # ── Load splits ────────────────────────────────────────────────────────
        def load_split(fname: str, has_label: bool) -> AIDOCellPerturbDataset:
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            symbols  = df["symbol"].tolist()
            pert_ids = df["pert_id"].tolist()
            gene_vocab_positions = [get_gene_pos(sym) for sym in symbols]
            labels = None
            if has_label and "label" in df.columns:
                rows = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return AIDOCellPerturbDataset(pert_ids, symbols, gene_vocab_positions, labels)

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

        # Store collate_fn builder result
        self._collate_fn = build_collate_fn(vocab_size=19264)

    def _loader(self, ds: AIDOCellPerturbDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Focal Loss with Class Weights ────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss with optional label smoothing and class weights.
    Input: logits [N, C], targets [N].

    class_weights [C] — per-class multiplier for up-weighting minority classes.
    These are applied before focal modulation to give minority classes a stronger
    gradient signal while focal modulation handles easy vs hard examples.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.reduction       = reduction
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)  # [N, C]

        if self.label_smoothing > 0.0:
            smooth_val  = self.label_smoothing / C
            one_hot     = torch.zeros_like(log_probs).scatter_(
                -1, targets.unsqueeze(-1), 1.0 - self.label_smoothing
            )
            one_hot    += smooth_val
            ce_per_sample = -(one_hot * log_probs).sum(dim=-1)  # [N]
            probs_true  = log_probs.gather(-1, targets.unsqueeze(-1)).exp().squeeze(-1)
        else:
            ce_per_sample = F.nll_loss(log_probs, targets, reduction="none")  # [N]
            probs_true    = log_probs.gather(-1, targets.unsqueeze(-1)).exp().squeeze(-1)

        focal_weight  = (1.0 - probs_true.detach()) ** self.gamma  # [N]
        loss          = focal_weight * ce_per_sample                # [N]

        # Apply per-sample class weight based on true class
        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]  # [N]
            loss = loss * sample_weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─── Enhanced Bilinear Interaction Head ───────────────────────────────────────

class EnhancedBilinearHead(nn.Module):
    """
    Enhanced bilinear interaction head with richer input representation.

    Key improvements over parent node2-2's BilinearHead:
    1. Input is a concatenation of two 640-dim representations:
       (a) gene_repr_pos: hidden state at perturbed gene's vocabulary position [B, 640]
       (b) gene_repr_pool: learnable attention-weighted mean over all 19,264 positions [B, 640]
       Combined input: [B, 1280]
    2. bilinear_dim increased from 256 to 512 — each of the 3 class subspaces has
       2x more capacity to represent interactions with 6,640 output gene embeddings.
    3. head_hidden_dim increased to 768 to handle the wider bilinear projection.

    Architecture:
      1. Input: concatenated [gene_repr_pos, gene_repr_pool] [B, 1280]
      2. MLP: LayerNorm(1280) -> Linear(1280, hidden_dim) -> GELU -> Dropout
              -> Linear(hidden_dim, n_classes * bilinear_dim) -> LayerNorm(proj_dim)
      3. Reshape: [B, n_classes, bilinear_dim]
      4. Bilinear: einsum("bcd,gd->bcg", proj, out_gene_emb) -> [B, n_classes, n_genes_out]

    Head parameter count:
      - Attention pooling: 640 weights (learnable)
      - MLP projector: 1280×768 + 768×1536 ≈ 2.16M
      - Output gene embeddings: 6640×512 ≈ 3.40M
      - Total: ~5.56M (vs 2.4M in parent — controlled increase, still far below node2-1's 42M)
    """

    def __init__(
        self,
        hidden_size: int  = HIDDEN_SIZE,        # 640 per representation (x2 = 1280 input)
        hidden_dim: int   = 768,
        bilinear_dim: int = 512,                 # increased from 256
        n_classes: int    = N_CLASSES,
        n_genes_out: int  = N_GENES_OUT,
        dropout: float    = 0.2,
        vocab_size: int   = 19264,              # for attention pooling
    ):
        super().__init__()
        input_size = hidden_size * 2  # concatenated: gene_pos + attention_pooled
        proj_dim   = n_classes * bilinear_dim   # 3 * 512 = 1536

        # Learnable attention weights for pooling over all gene positions
        # Initialized near-uniform so it starts close to mean-pool
        self.attn_pool_weights = nn.Parameter(torch.zeros(hidden_size))

        # MLP projection from concatenated representation to bilinear space
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # Learnable output gene embeddings [n_genes_out, bilinear_dim]
        self.out_gene_emb = nn.Embedding(n_genes_out, bilinear_dim)
        nn.init.normal_(self.out_gene_emb.weight, std=0.02)

        self.n_classes    = n_classes
        self.bilinear_dim = bilinear_dim
        self.n_genes_out  = n_genes_out

    def forward(
        self,
        gene_repr_pos: torch.Tensor,   # [B, hidden_size] — from perturbed gene's vocab position
        all_hidden: torch.Tensor,      # [B, seq_len, hidden_size] — all gene positions
    ) -> torch.Tensor:
        """
        Args:
            gene_repr_pos: [B, 640] — hidden state at perturbed gene's vocabulary position
            all_hidden: [B, 19266, 640] — full sequence hidden states (fused layers)
        Returns:
            logits: [B, n_classes, n_genes_out]
        """
        B = gene_repr_pos.shape[0]

        # Attention-weighted pooling over all positions
        # attn_pool_weights [640] determines a dot-product attention score for each position
        # (acts as a "query" vector for pooling over all 19,266 position "keys")
        attn_scores = torch.einsum(
            "bsd,d->bs", all_hidden, self.attn_pool_weights
        )  # [B, seq_len]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, seq_len]
        gene_repr_pool = torch.einsum(
            "bs,bsd->bd", attn_weights, all_hidden
        )  # [B, hidden_size]

        # Concatenate position-specific and global representations
        gene_repr = torch.cat([gene_repr_pos, gene_repr_pool], dim=-1)  # [B, 1280]

        # Project to bilinear space
        proj = self.mlp(gene_repr)                             # [B, n_classes * bilinear_dim]
        proj = proj.view(B, self.n_classes, self.bilinear_dim) # [B, 3, bilinear_dim]

        # Bilinear interaction: [B, 3, bilinear_dim] x [n_genes_out, bilinear_dim]^T
        # -> [B, 3, n_genes_out]
        out_emb = self.out_gene_emb.weight  # [n_genes_out, bilinear_dim]
        logits  = torch.einsum("bcd,gd->bcg", proj, out_emb)  # [B, 3, n_genes_out]
        return logits


# ─── Full Model ───────────────────────────────────────────────────────────────

class AIDOCellPerturbModel(nn.Module):
    """
    AIDO.Cell-100M + LoRA r=64 backbone with multi-layer fusion
    + enhanced bilinear head (bilinear_dim=512, dual representation input).
    """

    def __init__(
        self,
        lora_r: int          = 64,
        lora_alpha: int      = 128,
        lora_dropout: float  = 0.05,
        head_hidden_dim: int = 768,
        head_bilinear_dim: int = 512,
        head_dropout: float  = 0.2,
        n_genes_out: int     = N_GENES_OUT,
        n_classes: int       = N_CLASSES,
    ):
        super().__init__()

        # Load backbone
        backbone = AutoModel.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False

        # AIDO.Cell does not implement get_input_embeddings(), which PEFT calls
        # via enable_input_require_grads() unconditionally in PeftModel.__init__.
        # Patch enable_input_require_grads to register a hook on the gene embedding
        # layer instead, ensuring gradient flow through the frozen backbone.
        def _safe_enable_input_require_grads():
            def _make_inputs_require_grad(module, input, output):
                if isinstance(output, torch.Tensor):
                    output.requires_grad_(True)
            backbone.bert.gene_embedding.register_forward_hook(_make_inputs_require_grad)
        backbone.enable_input_require_grads = _safe_enable_input_require_grads

        # Apply LoRA to Q/K/V in ALL 18 layers
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Enable gradient checkpointing after PEFT wrapping for memory efficiency.
        self.backbone.base_model.model.config.use_cache = False
        self.backbone.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA params to float32 for optimizer stability
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Learnable layer-fusion weights for last FUSION_LAYERS layers
        self.layer_weights = nn.Parameter(torch.zeros(FUSION_LAYERS))

        # Enhanced bilinear interaction prediction head
        self.head = EnhancedBilinearHead(
            hidden_size=HIDDEN_SIZE,
            hidden_dim=head_hidden_dim,
            bilinear_dim=head_bilinear_dim,
            n_classes=n_classes,
            n_genes_out=n_genes_out,
            dropout=head_dropout,
            vocab_size=19264,
        )

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, 19264] float32
        gene_positions: torch.Tensor,  # [B] long – vocabulary index of perturbed gene
    ) -> torch.Tensor:
        """Returns logits [B, 3, 6640]."""
        attn_mask = torch.ones(
            input_ids.shape[0], input_ids.shape[1],
            dtype=torch.long, device=input_ids.device,
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )

        # hidden_states: tuple of 19 tensors [B, 19266, 640]
        # indices 0..18; we use the last FUSION_LAYERS = [13..18]
        hidden_states = torch.stack(
            [out.hidden_states[i].float()
             for i in range(N_LAYERS - FUSION_LAYERS + 1, N_LAYERS + 1)],
            dim=0,
        )  # [FUSION_LAYERS, B, 19266, 640]

        # Learnable weighted combination across fusion layers
        weights = torch.softmax(self.layer_weights, dim=0)  # [FUSION_LAYERS]
        fused   = (hidden_states * weights[:, None, None, None]).sum(0)  # [B, 19266, 640]

        # Extract the representation at each sample's perturbed gene's vocabulary position.
        # gene_positions are in [0, 19263]; AIDO.Cell output has 19266 positions
        # (19264 genes + 2 summary tokens), so gene vocabulary indices index correctly.
        B = fused.shape[0]
        gene_repr_pos = fused[torch.arange(B, device=fused.device), gene_positions, :]  # [B, 640]

        # Enhanced bilinear head uses both position-specific and global representations
        logits = self.head(gene_repr_pos, fused)  # [B, 3, 6640]
        return logits


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(local_p: torch.Tensor, local_l: torch.Tensor,
                    device, world_size: int):
    """All-gather variable-length tensors for DDP metric aggregation."""
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
        lr_backbone: float     = 5e-5,
        lr_head: float         = 3e-4,
        weight_decay: float    = 1e-3,
        focal_gamma: float     = 2.0,
        label_smoothing: float = 0.05,
        class_weight_down: float = 2.0,   # weight for down-regulated class (index 0)
        class_weight_neutral: float = 0.5, # weight for neutral class (index 1)
        class_weight_up: float = 4.0,     # weight for up-regulated class (index 2)
        lora_r: int            = 64,
        lora_alpha: int        = 128,
        head_hidden_dim: int   = 768,
        head_bilinear_dim: int = 512,
        head_dropout: float    = 0.2,
        warmup_steps: int      = 100,
        max_steps_total: int   = 3600,    # ~80 epochs * 45 steps/epoch (1416 samples, global_bs=32)
        gradient_clip_val: float = 1.0,
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
        self.model = AIDOCellPerturbModel(
            lora_r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            head_hidden_dim=self.hparams.head_hidden_dim,
            head_bilinear_dim=self.hparams.head_bilinear_dim,
            head_dropout=self.hparams.head_dropout,
        )

        # Mild class weights to improve minority class (up-regulated: ~3%) F1
        # weights: [down=2.0, neutral=0.5, up=4.0]
        cw = torch.tensor([
            self.hparams.class_weight_down,
            self.hparams.class_weight_neutral,
            self.hparams.class_weight_up,
        ], dtype=torch.float32)
        self.focal_loss = FocalLoss(
            gamma=self.hparams.focal_gamma,
            label_smoothing=self.hparams.label_smoothing,
            class_weights=cw,
        )

    def forward(self, input_ids, gene_positions):
        return self.model(input_ids, gene_positions)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Reshape to 2D before focal loss to use deterministic CUDA kernel."""
        # logits: [B, 3, 6640] -> [B*6640, 3];  labels: [B, 6640] -> [B*6640]
        logits_2d = logits.float().permute(0, 2, 1).reshape(-1, N_CLASSES)
        labels_1d = labels.reshape(-1)
        return self.focal_loss(logits_2d, labels_1d)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"])
        if "label" in batch:
            loss = self._loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True,
                     prog_bar=True, sync_dist=True)
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
            self.print(f"[Node2-2-1] Saved {len(dedup_indices)} test predictions -> {pred_path}")
            if all_labels.any():
                dedup_probs  = all_probs_np[dedup_indices]
                dedup_labels = all_labels[dedup_indices].numpy()
                f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
                self.print(f"[Node2-2-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Two parameter groups: backbone (LoRA + layer_weights) vs prediction head
        backbone_params = []
        head_params     = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("head.") or name == "layer_weights":
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": hp.lr_backbone},
                {"params": head_params,     "lr": hp.lr_head},
            ],
            weight_decay=hp.weight_decay,
        )

        # Cosine annealing with linear warmup, calibrated to ~80 effective epochs
        warmup = hp.warmup_steps
        total  = hp.max_steps_total

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total - warmup))
            progress = min(progress, 1.0)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and persistent buffers to minimize checkpoint size."""
        full_sd       = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        sd = {k: v for k, v in full_sd.items() if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving ckpt: {trained}/{total} params ({100*trained/total:.2f}%)"
        )
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Node 2-2-1 – AIDO.Cell-100M LoRA r=64 + Enhanced Bilinear Head (bilinear_dim=512)")
    p.add_argument("--data-dir",              type=str,   default="data")
    p.add_argument("--lr-backbone",           type=float, default=5e-5)
    p.add_argument("--lr-head",               type=float, default=3e-4)
    p.add_argument("--weight-decay",          type=float, default=1e-3)
    p.add_argument("--focal-gamma",           type=float, default=2.0)
    p.add_argument("--label-smoothing",       type=float, default=0.05)
    p.add_argument("--class-weight-down",     type=float, default=2.0,
                   help="Focal loss weight for down-regulated class (index 0)")
    p.add_argument("--class-weight-neutral",  type=float, default=0.5,
                   help="Focal loss weight for neutral class (index 1)")
    p.add_argument("--class-weight-up",       type=float, default=4.0,
                   help="Focal loss weight for up-regulated class (index 2)")
    p.add_argument("--lora-r",                type=int,   default=64)
    p.add_argument("--lora-alpha",            type=int,   default=128)
    p.add_argument("--head-hidden-dim",       type=int,   default=768)
    p.add_argument("--head-bilinear-dim",     type=int,   default=512)
    p.add_argument("--head-dropout",          type=float, default=0.2)
    p.add_argument("--warmup-steps",          type=int,   default=100)
    p.add_argument("--micro-batch-size",      type=int,   default=4)
    p.add_argument("--global-batch-size",     type=int,   default=32)
    p.add_argument("--max-epochs",            type=int,   default=150)
    p.add_argument("--patience",              type=int,   default=30)
    p.add_argument("--num-workers",           type=int,   default=2)
    p.add_argument("--gradient-clip-val",     type=float, default=1.0)
    p.add_argument("--val-check-interval",    type=float, default=1.0)
    p.add_argument("--debug-max-step",        type=int,   default=None,
                   help="Limit training/val/test to this many steps (debug only)")
    p.add_argument("--fast-dev-run",          action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Effective number of steps per epoch: ceil(1416 / global_batch_size)
    steps_per_epoch = int(np.ceil(1416 / args.global_batch_size))  # = 45 for bs=32
    # Calibrate cosine schedule to ~80 effective epochs (extended from parent's 60)
    # to allow a more gradual LR decay through the plateau region
    max_steps_total = steps_per_epoch * 80  # = 3600 steps ~ 80 epochs
    if args.debug_max_step is not None:
        max_steps_total = args.debug_max_step

    dm  = AIDOCellDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = AIDOCellLitModule(
        lr_backbone         = args.lr_backbone,
        lr_head             = args.lr_head,
        weight_decay        = args.weight_decay,
        focal_gamma         = args.focal_gamma,
        label_smoothing     = args.label_smoothing,
        class_weight_down   = args.class_weight_down,
        class_weight_neutral = args.class_weight_neutral,
        class_weight_up     = args.class_weight_up,
        lora_r              = args.lora_r,
        lora_alpha          = args.lora_alpha,
        head_hidden_dim     = args.head_hidden_dim,
        head_bilinear_dim   = args.head_bilinear_dim,
        head_dropout        = args.head_dropout,
        warmup_steps        = args.warmup_steps,
        max_steps_total     = max_steps_total,
        gradient_clip_val   = args.gradient_clip_val,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.patience, min_delta=1e-5,
    )
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps:   int | None = None
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

    accum    = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        accumulate_grad_batches=accum,
        gradient_clip_val=args.gradient_clip_val,
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

    trainer.fit(lit, datamodule=dm)

    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 2-2-1 – AIDO.Cell-100M LoRA r=64 + Enhanced Bilinear Head (bilinear_dim=512)\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
