"""
Node 3-3 – AIDO.Cell-100M + Realistic Multi-Gene Input + STRING_GNN Fusion + Deep Residual Head

Architecture:
  - AIDO.Cell-100M backbone (hidden_size=640, 18 transformer layers)
  - Realistic knockdown perturbation input:
      all 19,264 genes at expression=1.0 (baseline), perturbed gene at -1.0 (silenced/missing)
      This keeps AIDO.Cell in-distribution and allows it to leverage learned co-expression patterns.
  - LoRA fine-tuning (r=16, alpha=32) on Q/K/V matrices, pure AdamW
  - Gene-position extraction: hidden state at perturbed gene's vocab index
  - STRING_GNN (frozen, precomputed once): 256-dim PPI embeddings as second stream
  - Gated fusion of AIDO.Cell 640-dim + STRING_GNN 256-dim → 640-dim
  - 6-layer residual MLP head + bilinear interaction head (following node1-2's architecture)
  - Focal cross-entropy loss (gamma=2.0) + inverse-frequency class weights [10.91, 1.0, 29.62]
  - CosineAnnealingLR (T_max=100) with 10-epoch linear warmup

Key improvements over sibling node3-2:
  1. Realistic multi-gene input (in-distribution for AIDO.Cell; addresses primary bottleneck)
  2. STRING_GNN PPI stream (cross-perturbation diversity; adds complementary signal)
  3. Gated fusion (better synergy extraction than simple concatenation)
  4. Deeper 6-layer residual + bilinear head (proven in node1-2 lineage, F1=0.4912)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import argparse
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

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
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy, SingleDeviceStrategy
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ─── Constants ────────────────────────────────────────────────────────────────

AIDO_CELL_DIR  = "/home/Models/AIDO.Cell-100M"
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT    = 6640
N_CLASSES      = 3
HIDDEN_SIZE    = 640   # AIDO.Cell-100M hidden_size
GNN_HIDDEN     = 256   # STRING_GNN output dimension
N_GENE_VOCAB   = 19264 # AIDO.Cell gene vocabulary size
HEAD_DIM       = 512   # Residual MLP hidden dimension
HEAD_EXPAND    = 4     # MLP intermediate expansion factor
HEAD_RANK      = 512   # Bilinear output gene embedding rank
N_RESBLOCKS    = 6     # Number of residual MLP blocks

# Inverse-frequency class weights from training data distribution:
# down(-1): 8.14% → weight=88.86/8.14=10.91
# neutral(0): 88.86% → weight=1.0
# up(+1): 3.00% → weight=88.86/3.00=29.62
CLASS_WEIGHTS = torch.tensor([10.91, 1.0, 29.62], dtype=torch.float32)


# ─── Metric (matches calc_metric.py logic) ────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """
    Computes macro-averaged per-gene F1.
    pred_np: [N, 3, G] float — class probabilities (softmax)
    labels_np: [N, G] int — class indices in {0,1,2}
    """
    pred_cls = pred_np.argmax(axis=1)  # [N, G]
    f1_vals  = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]
        yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbDataset(Dataset):
    def __init__(
        self,
        pert_ids:       List[str],
        symbols:        List[str],
        input_ids:      torch.Tensor,  # [N, 19264] float32 (realistic knockdown profiles)
        gene_positions: torch.Tensor,  # [N] long (perturbed gene vocab index)
        gnn_indices:    torch.Tensor,  # [N] long (STRING_GNN node index, -1 if OOV)
        labels:         Optional[torch.Tensor] = None,
    ):
        self.pert_ids       = pert_ids
        self.symbols        = symbols
        self.input_ids      = input_ids
        self.gene_positions = gene_positions
        self.gnn_indices    = gnn_indices
        self.labels         = labels

    def __len__(self): return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":       self.pert_ids[idx],
            "symbol":        self.symbols[idx],
            "input_ids":     self.input_ids[idx],
            "gene_position": self.gene_positions[idx],
            "gnn_index":     self.gnn_indices[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    out = {
        "pert_id":       [b["pert_id"]       for b in batch],
        "symbol":        [b["symbol"]        for b in batch],
        "input_ids":     torch.stack([b["input_ids"]     for b in batch]),
        "gene_position": torch.stack([b["gene_position"] for b in batch]),
        "gnn_index":     torch.stack([b["gnn_index"]     for b in batch]),
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class DataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "data", micro_batch_size: int = 4, num_workers: int = 2):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage=None):
        # ── Load tokenizer (rank-0 downloads first, then all ranks load) ──
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)

        # ── STRING_GNN: build Ensembl ID → node index mapping ──
        gnn_model_dir = Path(STRING_GNN_DIR)
        node_names: List[str] = json.loads((gnn_model_dir / "node_names.json").read_text())
        ens_to_gnn: Dict[str, int] = {name: i for i, name in enumerate(node_names)}
        self.n_gnn_nodes = len(node_names)

        # ── Build realistic baseline input (all 19264 genes at expression=1.0) ──
        # AIDO.Cell's tokenizer has `gene_to_index` mapping gene symbols → vocab positions.
        # We can directly construct the all-1.0 baseline vector and modify the
        # perturbed gene's position to -1.0 (silenced/missing = knockdown simulation).
        # This is efficient: O(1) per sample, using the pre-built index.
        all_gene_names = tokenizer.gene_names  # List[str] of length 19264 in vocab order
        baseline_out = tokenizer(
            {"gene_names": all_gene_names, "expression": [1.0] * len(all_gene_names)},
            return_tensors="pt",
        )
        baseline_ids = baseline_out["input_ids"].squeeze(0).clone()  # [19264] float32
        # gene_to_index: Dict[str, int] mapping gene symbol → position in [19264] vector
        gene_to_idx: Dict[str, int] = tokenizer.gene_to_index

        def build_knockdown_inputs(
            symbols: List[str],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Build realistic knockdown perturbation inputs.
            For each sample:
              - Start from baseline (all genes at expression=1.0)
              - Set the perturbed gene's position to -1.0 (silenced = knockdown)
            Returns:
              input_ids_batch: [N, 19264] float32
              gene_positions:  [N] long  (perturbed gene's vocab index)
            """
            # Pre-allocate output tensor (N, 19264) — baseline values everywhere
            N = len(symbols)
            input_ids_batch = baseline_ids.unsqueeze(0).expand(N, -1).clone()  # [N, 19264]
            gene_positions  = torch.zeros(N, dtype=torch.long)

            for i, sym in enumerate(symbols):
                gpos = gene_to_idx.get(sym, None)
                if gpos is not None:
                    input_ids_batch[i, gpos] = -1.0  # Silence perturbed gene (knockdown)
                    gene_positions[i] = gpos
                else:
                    # Symbol not in AIDO.Cell vocabulary — use position 0 as fallback
                    # (rare case; baseline profile unchanged for this sample)
                    gene_positions[i] = 0

            return input_ids_batch, gene_positions

        def build_gnn_indices(pert_ids: List[str]) -> torch.Tensor:
            """Map Ensembl gene IDs to STRING_GNN node indices. Returns -1 for OOV genes."""
            indices = [ens_to_gnn.get(pid, -1) for pid in pert_ids]
            return torch.tensor(indices, dtype=torch.long)

        def load_split(fname: str, has_lbl: bool) -> PerturbDataset:
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            input_ids, gene_positions = build_knockdown_inputs(df["symbol"].tolist())
            gnn_indices               = build_gnn_indices(df["pert_id"].tolist())
            labels = None
            if has_lbl and "label" in df.columns:
                # Shift labels from {-1, 0, 1} to class indices {0, 1, 2}
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return PerturbDataset(
                pert_ids       = df["pert_id"].tolist(),
                symbols        = df["symbol"].tolist(),
                input_ids      = input_ids,
                gene_positions = gene_positions,
                gnn_indices    = gnn_indices,
                labels         = labels,
            )

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  False)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=(self.num_workers > 0),
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Model Components ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block (GELU activation, no bias)."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim * expand, bias=True)
        self.fc2  = nn.Linear(dim * expand, dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.norm(x)))))


class GatedFusion(nn.Module):
    """
    Gated element-wise fusion of two embedding streams.
    gate ∈ [0,1] controls how much of AIDO vs GNN to use per dimension.
    Output is LayerNorm-normalized.
    """

    def __init__(self, d_aido: int, d_gnn: int, d_out: int):
        super().__init__()
        d_cat         = d_aido + d_gnn
        self.proj_a   = nn.Linear(d_aido, d_out, bias=False)
        self.proj_g   = nn.Linear(d_gnn,  d_out, bias=False)
        self.gate_net = nn.Sequential(
            nn.Linear(d_cat, d_out, bias=True),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_out)

    def forward(self, aido: torch.Tensor, gnn: torch.Tensor) -> torch.Tensor:
        cat   = torch.cat([aido, gnn], dim=-1)          # [B, d_aido+d_gnn]
        gate  = self.gate_net(cat)                       # [B, d_out]
        pa    = self.proj_a(aido)                        # [B, d_out]
        pg    = self.proj_g(gnn)                         # [B, d_out]
        fused = gate * pa + (1.0 - gate) * pg            # [B, d_out]
        return self.norm(fused)


class Node33Model(nn.Module):
    """
    AIDO.Cell-100M + STRING_GNN fusion model.

    Pipeline:
      1. AIDO.Cell-100M (LoRA r=16): realistic knockdown input → gene-position extraction → [B, 640]
      2. STRING_GNN (frozen buffer): Ensembl ID lookup → [B, 256]
      3. Gated fusion: [B, 640] + [B, 256] → [B, 640]
      4. 6-layer residual MLP: [B, 640] → [B, head_dim=512]
      5. Bilinear output: [B, 512] × [6640, 512] → [B, 3, 6640]
    """

    def __init__(
        self,
        n_gnn_nodes:  int,
        n_genes_out:  int   = N_GENES_OUT,
        n_classes:    int   = N_CLASSES,
        head_dim:     int   = HEAD_DIM,
        head_expand:  int   = HEAD_EXPAND,
        head_rank:    int   = HEAD_RANK,
        n_resblocks:  int   = N_RESBLOCKS,
        head_dropout: float = 0.2,
        lora_r:       int   = 16,
        lora_alpha:   int   = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.n_gnn_nodes = n_gnn_nodes
        self.head_rank   = head_rank
        self.n_classes   = n_classes

        # ── AIDO.Cell-100M backbone with LoRA ──────────────────────────────
        backbone = AutoModel.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,  # All 18 transformer layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.config.use_cache = False
        # Enable gradient checkpointing on the underlying model (bypasses PEFT wrapper)
        self.backbone.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA params to float32 for stable optimization
        # (frozen backbone params remain in bfloat16 for memory efficiency)
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── STRING_GNN embeddings (frozen buffer, precomputed) ─────────────
        # Buffer is populated in LightningModule.setup() via load_string_gnn_embeddings()
        self.register_buffer("gnn_emb", torch.zeros(n_gnn_nodes, GNN_HIDDEN))
        # Learnable OOV embedding for genes not in STRING vocabulary
        self.oov_emb = nn.Parameter(torch.zeros(1, GNN_HIDDEN))
        nn.init.normal_(self.oov_emb, std=0.02)

        # ── Gated fusion: AIDO 640 + GNN 256 → 640 ────────────────────────
        self.fusion = GatedFusion(HIDDEN_SIZE, GNN_HIDDEN, HIDDEN_SIZE)

        # ── Projection into head dimension ─────────────────────────────────
        self.proj_in = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, head_dim),
            nn.LayerNorm(head_dim),
        )

        # ── 6-layer residual MLP ──────────────────────────────────────────
        self.resblocks = nn.ModuleList([
            ResidualBlock(head_dim, head_expand, head_dropout)
            for _ in range(n_resblocks)
        ])

        # ── Bilinear output interaction head ──────────────────────────────
        # Following node1-2's proven architecture:
        # pert_emb [B, n_classes, head_rank] × out_gene_emb [n_genes_out, head_rank]^T
        # → logits [B, n_classes, n_genes_out]
        self.proj_to_rank = nn.Linear(head_dim, n_classes * head_rank, bias=False)
        self.out_gene_emb = nn.Embedding(n_genes_out, head_rank)
        nn.init.normal_(self.out_gene_emb.weight, std=0.02)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"[Node3-3] Trainable params: {n_trainable:,} | Frozen params: {n_frozen:,}")

    def forward(
        self,
        input_ids:      torch.Tensor,  # [B, 19264] float32 (knockdown profile)
        gene_positions: torch.Tensor,  # [B] long  (vocab index of perturbed gene)
        gnn_indices:    torch.Tensor,  # [B] long  (-1 for OOV genes)
    ) -> torch.Tensor:
        B = input_ids.shape[0]

        # ── AIDO.Cell backbone forward ────────────────────────────────────
        attn_mask = torch.ones(B, N_GENE_VOCAB, dtype=torch.long, device=input_ids.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.backbone(input_ids=input_ids, attention_mask=attn_mask)

        # last_hidden_state: [B, 19266, 640] → slice gene positions only: [B, 19264, 640]
        gene_states = out.last_hidden_state[:, :N_GENE_VOCAB, :].float()  # [B, 19264, 640]

        # Gene-position extraction: extract the perturbed gene's contextual representation
        pos_idx  = gene_positions.view(B, 1, 1).expand(B, 1, HIDDEN_SIZE)  # [B, 1, 640]
        aido_emb = gene_states.gather(1, pos_idx).squeeze(1)               # [B, 640]

        # ── STRING_GNN embedding lookup ──────────────────────────────────
        oov_mask = (gnn_indices < 0)    # [B] bool
        safe_idx = gnn_indices.clone()
        safe_idx[oov_mask] = 0          # Prevent index-out-of-bounds; replaced below
        gnn_part = self.gnn_emb[safe_idx]              # [B, 256] float32
        # Replace OOV entries with learnable OOV embedding
        if oov_mask.any():
            oov_expanded = self.oov_emb.expand(oov_mask.sum(), -1)  # [n_oov, 256]
            gnn_part = gnn_part.clone()
            gnn_part[oov_mask] = oov_expanded.to(gnn_part.dtype)

        # ── Gated fusion: [B, 640] + [B, 256] → [B, 640] ─────────────────
        fused = self.fusion(aido_emb, gnn_part)        # [B, 640]

        # ── Residual MLP: [B, 640] → [B, head_dim] ───────────────────────
        h = self.proj_in(fused)                        # [B, 512]
        for block in self.resblocks:
            h = block(h)                               # [B, 512]

        # ── Bilinear output head: [B, 512] → [B, 3, 6640] ────────────────
        pert_flat = self.proj_to_rank(h)               # [B, 3 * head_rank]
        pert_3d   = pert_flat.view(B, self.n_classes, self.head_rank)  # [B, 3, rank]
        gene_emb  = self.out_gene_emb.weight           # [6640, rank]
        logits    = torch.einsum("bcr,gr->bcg", pert_3d, gene_emb)    # [B, 3, 6640]

        return logits


# ─── Focal Loss ───────────────────────────────────────────────────────────────

def focal_cross_entropy(
    logits:        torch.Tensor,   # [B, C, G]
    targets:       torch.Tensor,   # [B, G]  class indices in {0, 1, 2}
    class_weights: torch.Tensor,   # [C]
    gamma:         float = 2.0,
) -> torch.Tensor:
    """
    Focal cross-entropy with class weights.
    Computes per-position (B*G) focal-modulated cross-entropy and returns the mean.
    """
    B, C, G = logits.shape
    logits_flat  = logits.permute(0, 2, 1).reshape(B * G, C)   # [B*G, C]
    targets_flat = targets.reshape(B * G)                       # [B*G]

    # Cross-entropy per position (with class weighting)
    ce_loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        weight=class_weights.to(logits.device),
        reduction="none",
    )  # [B*G]

    # Focal modulation: (1 - p_t)^gamma
    with torch.no_grad():
        probs = torch.softmax(logits_flat, dim=1)                            # [B*G, C]
        p_t   = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)        # [B*G]
        focal_w = (1.0 - p_t) ** gamma                                       # [B*G]

    return (focal_w * ce_loss).mean()


# ─── DDP gather helper ────────────────────────────────────────────────────────

def _gather_tensors(
    local_p:    torch.Tensor,
    local_l:    torch.Tensor,
    device:     torch.device,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather tensors from all DDP ranks, padding shorter ranks to max size."""
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))
    pad    = max_sz - local_p.shape[0]
    p = local_p.to(device); l = local_l.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], 0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], 0)
    gp = [torch.zeros_like(p) for _ in range(world_size)]
    gl = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(gp, p); dist.all_gather(gl, l)
    rp = torch.cat([gp[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    rl = torch.cat([gl[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    return rp, rl


# ─── LightningModule ──────────────────────────────────────────────────────────

class Node33LitModule(pl.LightningModule):

    def __init__(
        self,
        n_gnn_nodes:   int   = 18870,
        lr_backbone:   float = 1e-4,
        lr_head:       float = 3e-4,
        weight_decay:  float = 0.01,
        focal_gamma:   float = 2.0,
        lora_r:        int   = 16,
        lora_alpha:    int   = 32,
        lora_dropout:  float = 0.05,
        head_dropout:  float = 0.2,
        warmup_epochs: int   = 10,
        t_max:         int   = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []
        self._test_labels:   List[torch.Tensor] = []

    def setup(self, stage=None):
        hp = self.hparams
        self.model = Node33Model(
            n_gnn_nodes  = hp.n_gnn_nodes,
            head_dropout = hp.head_dropout,
            lora_r       = hp.lora_r,
            lora_alpha   = hp.lora_alpha,
            lora_dropout = hp.lora_dropout,
        )
        self.register_buffer("class_weights", CLASS_WEIGHTS)

        # ── Load STRING_GNN embeddings into model buffer ──────────────────
        # Load on all ranks (frozen, read-only — no coordination needed)
        gnn_emb = self._load_string_gnn_embeddings()
        self.model.gnn_emb.copy_(gnn_emb)
        self.model.gnn_emb.requires_grad_(False)

    @staticmethod
    def _load_string_gnn_embeddings() -> torch.Tensor:
        """
        Run frozen STRING_GNN forward pass to get precomputed node embeddings [18870, 256].
        Returns float32 tensor on CPU.
        """
        model_dir = Path(STRING_GNN_DIR)
        gnn_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        gnn_model.eval()

        graph       = torch.load(model_dir / "graph_data.pt", weights_only=False)
        edge_index  = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)

        with torch.no_grad():
            outputs = gnn_model(edge_index=edge_index, edge_weight=edge_weight)
            gnn_emb = outputs.last_hidden_state.float()  # [18870, 256]

        del gnn_model
        return gnn_emb

    def forward(self, input_ids, gene_positions, gnn_indices):
        return self.model(input_ids, gene_positions, gnn_indices)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return focal_cross_entropy(
            logits, labels,
            class_weights=self.class_weights,
            gamma=self.hparams.focal_gamma,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"], batch["gnn_index"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"], batch["gnn_index"])
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
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"], batch["gnn_index"])
        probs  = torch.softmax(logits, dim=1)   # [B, 3, 6640]
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs  = torch.cat(self._test_preds, 0)
        dummy_labels = (
            torch.cat(self._test_labels, 0) if self._test_labels
            else torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        )

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

            # Deduplicate by pert_id (DDP DistributedSampler may pad the dataset)
            seen_pids: set = set()
            dedup_perts, dedup_syms, dedup_probs_list, dedup_label_rows = [], [], [], []
            for pid, sym, prob_row, lbl_row in zip(
                all_pert, all_syms,
                all_probs.numpy(), all_labels.numpy()
            ):
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    dedup_perts.append(pid)
                    dedup_syms.append(sym)
                    dedup_probs_list.append(prob_row)
                    dedup_label_rows.append(lbl_row)

            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for pid, sym, probs in zip(dedup_perts, dedup_syms, dedup_probs_list):
                    fh.write(f"{pid}\t{sym}\t{json.dumps(probs.tolist())}\n")

            self.print(f"[Node3-3] Saved {len(dedup_perts)} test predictions → {pred_path}")

            if self._test_labels:
                dedup_probs_np  = np.array(dedup_probs_list)
                dedup_labels_np = np.array(dedup_label_rows)
                f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                self.print(f"[Node3-3] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear(); self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Group 1: LoRA backbone parameters (lower LR to preserve pretrained representations)
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        backbone_ids    = {id(p) for p in backbone_params}
        # Group 2: All other trainable parameters (fusion, residual MLP, bilinear head)
        head_params     = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in backbone_ids
        ]

        param_groups = [
            {"params": backbone_params, "lr": hp.lr_backbone, "weight_decay": hp.weight_decay},
            {"params": head_params,     "lr": hp.lr_head,     "weight_decay": hp.weight_decay},
        ]
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

        # LR schedule: linear warmup then cosine annealing
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_epochs,
        )
        cosine_t_max     = max(1, hp.t_max - hp.warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[hp.warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and registered buffers (excludes frozen backbone)."""
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
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
        description="Node 3-3: AIDO.Cell-100M + Realistic Multi-Gene Input + STRING_GNN Fusion"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--lr-backbone",        type=float, default=1e-4)
    p.add_argument("--lr-head",            type=float, default=3e-4)
    p.add_argument("--weight-decay",       type=float, default=0.01)
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--lora-r",             type=int,   default=16)
    p.add_argument("--lora-alpha",         type=int,   default=32)
    p.add_argument("--lora-dropout",       type=float, default=0.05)
    p.add_argument("--head-dropout",       type=float, default=0.2)
    p.add_argument("--warmup-epochs",      type=int,   default=10)
    p.add_argument("--t-max",              type=int,   default=100)
    p.add_argument("--micro-batch-size",   type=int,   default=4)
    p.add_argument("--global-batch-size",  type=int,   default=32)
    p.add_argument("--max-epochs",         type=int,   default=200)
    p.add_argument("--patience",           type=int,   default=40)
    p.add_argument("--num-workers",        type=int,   default=2)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",     type=int,   default=None)
    p.add_argument("--fast-dev-run",       action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine STRING_GNN node count for model initialization
    n_gnn_nodes = len(
        json.loads((Path(STRING_GNN_DIR) / "node_names.json").read_text())
    )

    dm  = DataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = Node33LitModule(
        n_gnn_nodes   = n_gnn_nodes,
        lr_backbone   = args.lr_backbone,
        lr_head       = args.lr_head,
        weight_decay  = args.weight_decay,
        focal_gamma   = args.focal_gamma,
        lora_r        = args.lora_r,
        lora_alpha    = args.lora_alpha,
        lora_dropout  = args.lora_dropout,
        head_dropout  = args.head_dropout,
        warmup_epochs = args.warmup_epochs,
        t_max         = args.t_max,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-5)
    lr_cb  = LearningRateMonitor(logging_interval="epoch")
    pb_cb  = TQDMProgressBar(refresh_rate=10)
    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps:   int          = -1
    limit_train: float | int  = 1.0
    limit_val:   float | int  = 1.0
    limit_test:  float | int  = 1.0
    fast_dev_run: bool        = False

    if args.debug_max_step is not None:
        max_steps    = args.debug_max_step
        limit_train  = args.debug_max_step
        limit_val    = 2
        limit_test   = 2
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
        max_steps=max_steps,
        accumulate_grad_batches=accum,
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
            "Node 3-3: AIDO.Cell-100M + Realistic Multi-Gene Input + STRING_GNN Fusion + "
            "Deep Residual Head\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
