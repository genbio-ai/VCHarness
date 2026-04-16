"""
Node 3-1-1-1 — AIDO.Cell-100M + LoRA rank=16 + STRING_GNN Bimodal Fusion

Architecture:
  - AIDO.Cell-100M backbone (hidden_size=640, 18 transformer layers)
    loaded from /home/Models/AIDO.Cell-100M
  - Sparse perturbation input: {perturbed_gene_symbol: 1.0, others: -1.0}
  - LoRA fine-tuning (rank=16, alpha=32) on QKV + FFN matrices
    → ~10.8M trainable params (expanded from parent's ~2.7M rank=8 QKV-only)
    → addresses the post-epoch-12 plateau seen in parent node3-1-1
  - Gene-position extraction: hidden_state[:, gene_pos_idx, :] → [B, 640]
  - STRING_GNN perturbation-conditioned embedding: 256-dim per perturbed gene
    → cond_emb injects perturbation identity into GNN forward pass for
      perturbation-specific PPI topology signal
  - Bimodal fusion: concat([640-dim AIDO.Cell, 256-dim STRING_GNN]) → [B, 896]
  - Prediction head: Linear(896→1536) + GELU + LayerNorm + Dropout(0.25)
                   + Linear(1536 → 6640×3)
  - Focal cross-entropy loss (gamma=2.0) with inverse-frequency class weights
  - Cosine annealing LR with linear warmup (10 epochs) for stable early training

Key improvements vs parent node3-1-1:
  1. LoRA rank=16 (from 8): doubles expressiveness, addresses early F1 plateau
  2. LoRA on FFN layers too: up_proj, down_proj, gate_proj (~10.8M vs 2.7M params)
  3. STRING_GNN cond_emb bimodal fusion: adds 256-dim PPI topology signal
     orthogonal to AIDO.Cell's transcriptomic context
  4. Linear warmup (10 epochs) before cosine decay: stabilizes early LoRA/GNN training
  5. Head dropout 0.25 (from 0.2): stronger regularization with larger param count

Root cause from parent: LoRA rank=8 constrained optimization to 8-dim subspace
of 640×640 weight space → plateau at val F1~0.41 from epoch 12 onwards.
STRING_GNN adds orthogonal PPI topology signal that AIDO.Cell-100M cannot capture
from the sparse single-gene input.
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

AIDO_CELL_DIR = "/home/Models/AIDO.Cell-100M"
STRING_GNN_DIR = "/home/Models/STRING_GNN"

N_GENES_OUT  = 6640
N_CLASSES    = 3
HIDDEN_SIZE  = 640   # AIDO.Cell-100M hidden size
GNN_DIM      = 256   # STRING_GNN hidden size
FUSION_DIM   = HIDDEN_SIZE + GNN_DIM  # 896 = 640 + 256
N_GENE_VOCAB = 19264  # AIDO.Cell gene space

# Class weights: inverse-frequency based on train split label distribution
# down-regulated (-1): 8.14%, neutral (0): 88.86%, up-regulated (+1): 3.00%
# Shifted to {0,1,2}: class 0 = down, class 1 = neutral, class 2 = up
# Weights: neutral_freq / class_freq (normalized so neutral = 1.0)
CLASS_WEIGHTS = torch.tensor([10.91, 1.0, 29.62], dtype=torch.float32)


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """
    Compute macro-averaged per-gene F1 score matching calc_metric.py logic.

    Args:
        pred_np: [N, 3, G] softmax probabilities (float)
        labels_np: [N, G] class indices in {0, 1, 2} (already shifted from {-1, 0, 1})
    Returns:
        float: mean per-gene macro-F1 over all G genes
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


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal cross-entropy loss for multi-class classification.

    Focal loss down-weights well-classified examples and focuses training on
    hard examples. Particularly useful for the 88.9% neutral class imbalance
    in the DEG prediction task, where the model easily learns the neutral class
    but struggles with rare down/up-regulated positions.

    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    Args:
        gamma: focusing parameter (0 = standard CE, 2 = typical focal)
        weight: per-class weights tensor (same as CE weight parameter)
        label_smoothing: label smoothing factor
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, G] unnormalized logits (C=3 classes, G=6640 genes)
            targets: [B, G] class indices in {0, 1, 2}
        Returns:
            scalar loss
        """
        # [B, C, G] → [B*G, C]
        B, C, G = logits.shape
        logits_flat  = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, C]
        targets_flat = targets.reshape(-1)                      # [B*G]

        # Log-softmax probabilities for focal weight computation
        log_probs = F.log_softmax(logits_flat, dim=1)           # [B*G, C]
        probs     = torch.exp(log_probs)                        # [B*G, C]

        # Gather log-prob and prob at target class
        target_log_prob = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)  # [B*G]
        target_prob     = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)       # [B*G]

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1.0 - target_prob).pow(self.gamma)   # [B*G]

        # Per-class weight
        if self.weight is not None:
            class_w = self.weight.to(logits.device)[targets_flat]   # [B*G]
        else:
            class_w = torch.ones_like(focal_weight)

        # Label smoothing: blend target log-prob with mean log-prob
        if self.label_smoothing > 0:
            smooth_loss  = -log_probs.mean(dim=1)                    # [B*G]
            ce_loss      = -target_log_prob                          # [B*G]
            loss_per_pos = (
                (1 - self.label_smoothing) * ce_loss
                + self.label_smoothing * smooth_loss
            )
        else:
            loss_per_pos = -target_log_prob                         # [B*G]

        # Apply focal weighting and class weights
        weighted_loss = focal_weight * class_w * loss_per_pos       # [B*G]

        # Normalize by sum of weights for scale consistency
        denom = class_w.sum().clamp(min=1.0)
        return (weighted_loss.sum() / denom)


# ─── STRING_GNN Embedding Cache ────────────────────────────────────────────────

def build_string_gnn_embeddings_dict(
    pert_ids: List[str],
    device: torch.device,
) -> Dict[str, Tuple[torch.Tensor, bool]]:
    """
    Build STRING_GNN embeddings for a list of unique pert_ids using cond_emb injection.

    For each unique pert_id:
    1. Find the node index in STRING_GNN's node_names.json
    2. Create a cond_emb with 1.0 at the perturbed node, 0.0 elsewhere
    3. Run STRING_GNN forward with cond_emb → [N_nodes, 256]
    4. Extract the perturbed gene's embedding row → [256]

    Returns a dict: {pert_id: (emb_256d_float32_cpu, in_vocab_bool)}
    This allows O(1) lookup when assembling per-split datasets.
    """
    model_dir   = Path(STRING_GNN_DIR)
    node_names  = json.loads((model_dir / "node_names.json").read_text())
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    n_nodes     = len(node_names)

    gnn_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    gnn_model = gnn_model.to(device)
    gnn_model.eval()

    graph = torch.load(model_dir / "graph_data.pt", weights_only=False)
    edge_index  = graph["edge_index"].to(device)
    edge_weight = graph["edge_weight"]
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    # Deduplicate: compute only for unique pert_ids to avoid redundant GNN passes
    unique_pids = list(set(pert_ids))
    result: Dict[str, Tuple[torch.Tensor, bool]] = {}

    with torch.no_grad():
        for pid in unique_pids:
            pid_clean = pid.split(".")[0]  # strip version suffix if present
            if pid_clean in name_to_idx:
                node_idx = name_to_idx[pid_clean]
                # Perturbation-conditioned forward: inject identity signal at perturbed node
                # → encodes how the perturbation propagates through the PPI graph
                cond = torch.zeros(n_nodes, GNN_DIM, device=device)
                cond[node_idx] = 1.0
                out = gnn_model(
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    cond_emb=cond,
                )
                emb = out.last_hidden_state[node_idx].cpu().float()
                result[pid] = (emb, True)
            else:
                # OOV gene: zero vector (will be replaced by learnable fallback in model)
                result[pid] = (torch.zeros(GNN_DIM), False)

    # Clean up GNN model from GPU to free memory before AIDO.Cell loads
    del gnn_model
    torch.cuda.empty_cache()

    return result


# ─── Dataset ──────────────────────────────────────────────────────────────────

class FusionPerturbDataset(Dataset):
    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        input_ids: torch.Tensor,       # [N, 19264] float32
        gene_positions: torch.Tensor,  # [N] long
        gnn_embs: torch.Tensor,        # [N, 256] float32
        in_vocab: torch.Tensor,        # [N] bool
        labels: Optional[torch.Tensor] = None,  # [N, 6640] long or None
    ):
        self.pert_ids      = pert_ids
        self.symbols       = symbols
        self.input_ids     = input_ids
        self.gene_positions = gene_positions
        self.gnn_embs      = gnn_embs
        self.in_vocab      = in_vocab
        self.labels        = labels

    def __len__(self): return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":       self.pert_ids[idx],
            "symbol":        self.symbols[idx],
            "input_ids":     self.input_ids[idx],
            "gene_position": self.gene_positions[idx],
            "gnn_emb":       self.gnn_embs[idx],
            "in_vocab":      self.in_vocab[idx],
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
        "gnn_emb":       torch.stack([b["gnn_emb"]       for b in batch]),
        "in_vocab":      torch.stack([b["in_vocab"]      for b in batch]),
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class FusionDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data", micro_batch_size=4, num_workers=2):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage=None):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # ── Tokenizer: rank 0 downloads first, then barrier, then all ranks load ─
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)

        # ── Load pre-computed GNN embedding cache ──────────────────────────────
        # The cache is pre-computed in main() BEFORE DDP initialization,
        # so no distributed barrier is needed here — the file already exists.
        cache_path = Path(__file__).parent / "run" / "gnn_emb_cache.pt"
        gnn_cache: Dict[str, Tuple[torch.Tensor, bool]] = torch.load(
            cache_path, weights_only=False
        )

        # ── Helper: tokenize gene symbols ─────────────────────────────────────
        def tokenize_symbols(symbols):
            batch_input = [{"gene_names": [s], "expression": [1.0]} for s in symbols]
            tok_out     = tokenizer(batch_input, return_tensors="pt")
            ids  = tok_out["input_ids"]          # [N, 19264] float32
            gpos = (ids > 0.5).float().argmax(dim=1).long()  # position of perturbed gene
            return ids, gpos

        # ── Helper: assemble dataset for one split ────────────────────────────
        def load_split(fname, has_lbl):
            df  = pd.read_csv(self.data_dir / fname, sep="\t")
            ids, gpos = tokenize_symbols(df["symbol"].tolist())
            labels = None
            if has_lbl and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)

            # Look up precomputed GNN embeddings from cache (O(1) per sample)
            gnn_embs_list  = []
            in_vocab_list  = []
            for pid in df["pert_id"].tolist():
                emb, in_v = gnn_cache.get(pid, (torch.zeros(GNN_DIM), False))
                gnn_embs_list.append(emb)
                in_vocab_list.append(in_v)

            gnn_embs = torch.stack(gnn_embs_list, dim=0)   # [N, 256]
            in_vocab = torch.tensor(in_vocab_list, dtype=torch.bool)

            return FusionPerturbDataset(
                df["pert_id"].tolist(), df["symbol"].tolist(),
                ids, gpos, gnn_embs, in_vocab, labels
            )

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds, batch_size=self.micro_batch_size, shuffle=shuffle,
            collate_fn=collate_fn, num_workers=self.num_workers,
            pin_memory=True, drop_last=shuffle,
            persistent_workers=self.num_workers > 0
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Bimodal Fusion Model ─────────────────────────────────────────────────────

class BimodalFusionModel(nn.Module):
    """
    AIDO.Cell-100M + STRING_GNN Bimodal Fusion.

    Architecture:
    1. AIDO.Cell-100M with LoRA rank=16 on QKV + FFN layers
       → gene-position extraction: hidden_state[:, gene_pos_idx, :] → [B, 640]
    2. STRING_GNN perturbation-conditioned embeddings [B, 256] (precomputed)
       → OOV handling: replace zero-vectors with learnable fallback embedding
    3. Fusion: concat([640, 256]) → [B, 896]
    4. Prediction head: Linear(896→1536) + GELU + LayerNorm + Dropout(0.25)
                      + Linear(1536→6640×3)

    Key design decisions:
    - LoRA rank=16 (vs parent's 8): doubles the expressiveness of the LoRA subspace
      to address the observed early plateau (best epoch 12, then flat for 30 epochs)
    - FFN LoRA: expands fine-tuning to up_proj/down_proj/gate_proj for richer
      token-level transformations while maintaining LoRA's regularization structure
    - STRING_GNN cond_emb fusion: adds PPI topology context orthogonal to
      AIDO.Cell's transcriptomic representations
    - OOV embedding: learnable 256-dim vector for genes not in STRING_GNN vocabulary
    - Larger head (1536 vs 2048 width) scales to the 896-dim fusion representation
    """

    def __init__(self, n_genes_out: int = N_GENES_OUT, n_classes: int = N_CLASSES):
        super().__init__()

        # ── AIDO.Cell-100M backbone ──────────────────────────────────────────
        backbone = AutoModel.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False

        # LoRA on QKV + FFN layers (rank=16 for expanded expressiveness)
        # target_modules includes QKV attention matrices and all 3 FFN projections
        # across all 18 transformer layers.
        # - rank=16: doubles the optimization subspace vs parent's rank=8
        # - alpha=32: standard 2×rank scaling
        # - FFN modules: up_proj, down_proj, gate_proj (SwiGLU FFN)
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["query", "key", "value", "up_proj", "down_proj", "gate_proj"],
            layers_to_transform=None,  # fine-tune all 18 layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)

        # Enable gradient checkpointing AFTER LoRA wrapping (per AIDO.Cell skill docs)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA params to float32 for training stability
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── STRING_GNN OOV fallback embedding ───────────────────────────────
        # Learnable 256-dim embedding for genes not in STRING_GNN vocabulary
        # (~6.4% of genes are OOV based on node1-2 coverage analysis)
        self.oov_embedding = nn.Parameter(torch.zeros(GNN_DIM, dtype=torch.float32))
        nn.init.normal_(self.oov_embedding, std=0.02)

        # GNN input normalization: normalize the precomputed 256-dim embeddings
        # before fusion to ensure consistent scale with AIDO.Cell representations
        self.gnn_norm = nn.LayerNorm(GNN_DIM)

        # ── Prediction head (bimodal fusion 896-dim → 6640×3) ────────────────
        # Processes the concatenation of AIDO.Cell (640-dim) + STRING_GNN (256-dim)
        # Note: Dropout 0.25 (increased from parent's 0.2) for stronger regularization
        # given the expanded parameter count of rank=16 + FFN LoRA
        self.head = nn.Sequential(
            nn.Linear(FUSION_DIM, 1536),
            nn.GELU(),
            nn.LayerNorm(1536),
            nn.Dropout(0.25),
            nn.Linear(1536, n_genes_out * n_classes),
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f"[Node3-1-1-1] Trainable params: {n_trainable:,} / {n_total:,} "
              f"({100*n_trainable/n_total:.2f}%)")

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, 19264] float32
        gene_positions: torch.Tensor,  # [B] long
        gnn_emb: torch.Tensor,         # [B, 256] float32 (precomputed STRING_GNN)
        in_vocab: torch.Tensor,        # [B] bool
    ) -> torch.Tensor:

        # ── AIDO.Cell forward (bfloat16) ──────────────────────────────────
        attn_mask = torch.ones(
            input_ids.shape[0], input_ids.shape[1],
            dtype=torch.long, device=input_ids.device
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.backbone(input_ids=input_ids, attention_mask=attn_mask)

        # last_hidden_state: [B, 19266, 640]
        # Slice to gene positions only (exclude 2 appended summary tokens)
        gene_states = out.last_hidden_state[:, :N_GENE_VOCAB, :].float()  # [B, 19264, 640]

        # Gene-position extraction: extract the perturbed gene's position embedding
        batch_size = gene_states.shape[0]
        pos_idx    = gene_positions.view(batch_size, 1, 1).expand(batch_size, 1, HIDDEN_SIZE)
        gene_repr  = gene_states.gather(1, pos_idx).squeeze(1)  # [B, 640]

        # ── STRING_GNN embedding handling ─────────────────────────────────
        # OOV handling: replace zero vectors for OOV genes with learnable fallback
        gnn_emb_device = gnn_emb.to(gene_repr.device)
        in_v = in_vocab.to(gene_repr.device)  # [B] bool
        oov_mask = ~in_v  # True where gene is OOV

        # Replace OOV rows with learnable embedding
        oov_fill = self.oov_embedding.unsqueeze(0).expand(batch_size, -1)  # [B, 256]
        gnn_final = torch.where(
            in_v.unsqueeze(1).expand_as(gnn_emb_device),
            gnn_emb_device,
            oov_fill,
        )

        # Normalize GNN embeddings
        gnn_normalized = self.gnn_norm(gnn_final)  # [B, 256]

        # ── Bimodal fusion and prediction ─────────────────────────────────
        fused = torch.cat([gene_repr, gnn_normalized], dim=1)  # [B, 896]

        logits = self.head(fused)                           # [B, 6640*3]
        return logits.view(-1, N_CLASSES, N_GENES_OUT)      # [B, 3, 6640]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
    """Gather tensors from all DDP ranks, handling variable-size padding."""
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

class BimodalFusionLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_backbone: float     = 5e-5,    # LoRA adapter learning rate (reduced for wider rank)
        lr_head: float         = 2e-4,    # Prediction head learning rate
        weight_decay: float    = 0.01,
        focal_gamma: float     = 2.0,
        label_smoothing: float = 0.05,
        max_epochs: int        = 200,
        t_max_cosine: int      = 100,
        warmup_epochs: int     = 10,      # Linear warmup before cosine decay
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
        self.model = BimodalFusionModel()
        self.register_buffer("class_weights", CLASS_WEIGHTS)
        self.focal_loss = FocalLoss(
            gamma=self.hparams.focal_gamma,
            weight=CLASS_WEIGHTS,
            label_smoothing=self.hparams.label_smoothing,
        )

    def forward(self, input_ids, gene_positions, gnn_emb, in_vocab):
        return self.model(input_ids, gene_positions, gnn_emb, in_vocab)

    def _loss(self, logits, labels):
        return self.focal_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        logits = self(
            batch["input_ids"], batch["gene_position"],
            batch["gnn_emb"], batch["in_vocab"]
        )
        loss = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(
            batch["input_ids"], batch["gene_position"],
            batch["gnn_emb"], batch["in_vocab"]
        )
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
        probs_np  = torch.softmax(lp, dim=1).numpy()
        labels_np = ll.numpy()
        f1 = compute_per_gene_f1(probs_np, labels_np)
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(
            batch["input_ids"], batch["gene_position"],
            batch["gnn_emb"], batch["in_vocab"]
        )
        probs  = torch.softmax(logits, dim=1)  # [B, 3, G]
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
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate by pert_id (DDP padding may create duplicates)
            seen_pids: set = set()
            dedup_perts, dedup_syms, dedup_probs_list, dedup_label_rows = [], [], [], []
            for pid, sym, prob_row, lbl_row in zip(
                all_pert, all_syms, all_probs.numpy(), all_labels.numpy()
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

            self.print(f"[Node3-1-1-1] Saved {len(dedup_perts)} test predictions → {pred_path}")

            if all_labels.any():
                dedup_probs_np  = np.array(dedup_probs_list)
                dedup_labels_np = np.array(dedup_label_rows)
                f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                self.print(f"[Node3-1-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    # ── Optimizer with linear warmup + cosine annealing ───────────────────────

    def configure_optimizers(self):
        hp = self.hparams

        # Three parameter groups:
        # 1. LoRA backbone params → lower LR (fine-tuning regime)
        # 2. Head + GNN normalization + OOV embedding → higher LR (fresh init)
        lora_params = [
            p for n, p in self.model.backbone.named_parameters()
            if p.requires_grad
        ]
        lora_ids = {id(p) for p in lora_params}

        head_and_extra_params = [
            p for n, p in self.named_parameters()
            if p.requires_grad and id(p) not in lora_ids
        ]

        param_groups = [
            {"params": lora_params,          "lr": hp.lr_backbone, "weight_decay": hp.weight_decay},
            {"params": head_and_extra_params, "lr": hp.lr_head,     "weight_decay": hp.weight_decay},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Linear warmup for `warmup_epochs` epochs, then cosine annealing.
        # Warmup stabilizes LoRA + GNN fusion early training dynamics.
        # Parent's best epoch was 12 (very fast convergence) — warmup helps
        # the model find a better initialization before the cosine decay phase.
        def lr_lambda(epoch):
            if epoch < hp.warmup_epochs:
                return float(epoch + 1) / float(hp.warmup_epochs)
            # Cosine decay from 1.0 to eta_min/lr ratio after warmup
            progress = (epoch - hp.warmup_epochs) / float(
                max(1, hp.t_max_cosine - hp.warmup_epochs)
            )
            cosine_val = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)))
            eta_min_ratio = 1e-6 / hp.lr_backbone
            return float(eta_min_ratio + (1.0 - eta_min_ratio) * cosine_val)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable parameters + buffers ─────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        sd = {k: v for k, v in full_sd.items()
              if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving ckpt: {trained:,}/{total:,} params ({100*trained/total:.2f}%)"
        )
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 3-1-1-1 – AIDO.Cell-100M + LoRA r=16 + STRING_GNN Bimodal Fusion"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--lr-backbone",        type=float, default=5e-5)
    p.add_argument("--lr-head",            type=float, default=2e-4)
    p.add_argument("--weight-decay",       type=float, default=0.01)
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--label-smoothing",    type=float, default=0.05)
    p.add_argument("--micro-batch-size",   type=int,   default=4)
    p.add_argument("--global-batch-size",  type=int,   default=32)
    p.add_argument("--max-epochs",         type=int,   default=200)
    p.add_argument("--t-max-cosine",       type=int,   default=100)
    p.add_argument("--warmup-epochs",      type=int,   default=10)
    p.add_argument("--patience",           type=int,   default=35)
    p.add_argument("--num-workers",        type=int,   default=2)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",     type=int,   default=None)
    p.add_argument("--fast-dev-run",       action="store_true", default=False)
    return p.parse_args()


def _precompute_gnn_cache(args):
    """
    Pre-compute STRING_GNN embeddings BEFORE DDP initialization.

    torchrun launches all ranks simultaneously but DDP is not initialized until
    Trainer.fit() is called. We exploit this window to compute the GNN embedding
    cache on rank 0 only, then use file-based polling so other ranks wait.

    This avoids the NCCL watchdog timeout that occurs when rank 0 is still
    computing embeddings while rank 1 is blocked at dist.barrier() inside setup().
    """
    import time as _time

    rank      = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cache_path = Path(__file__).parent / "run" / "gnn_emb_cache.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path = cache_path.with_suffix(".pt.ready")

    if not sentinel_path.exists():
        if rank == 0:
            print("[Pre-compute] Building STRING_GNN cond_emb embeddings "
                  "(rank 0, one-time)...", flush=True)
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
            data_dir = Path(args.data_dir)
            all_pids = list({
                pid
                for fname in ["train.tsv", "val.tsv", "test.tsv"]
                for pid in pd.read_csv(data_dir / fname, sep="\t")["pert_id"].tolist()
            })
            gnn_cache = build_string_gnn_embeddings_dict(all_pids, device)
            # Write to temp file, then rename atomically to avoid partial reads
            tmp_path = cache_path.with_suffix(".pt.tmp")
            torch.save(gnn_cache, tmp_path)
            tmp_path.rename(cache_path)
            # Create sentinel file so other ranks stop polling
            sentinel_path.touch()
            in_vocab_count = sum(1 for (_, iv) in gnn_cache.values() if iv)
            print(f"[Pre-compute] Cached {len(gnn_cache)} embeddings "
                  f"({in_vocab_count} in-vocab) → {cache_path}", flush=True)
        else:
            # Non-rank-0: poll for sentinel file (no DDP barrier needed yet)
            print(f"[Rank {rank}] Waiting for STRING_GNN cache from rank 0...", flush=True)
            while not sentinel_path.exists():
                _time.sleep(3)
            print(f"[Rank {rank}] STRING_GNN cache ready.", flush=True)


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute STRING_GNN embeddings before DDP init to avoid NCCL timeout
    _precompute_gnn_cache(args)

    dm  = FusionDataModule(
        args.data_dir, args.micro_batch_size, args.num_workers
    )
    lit = BimodalFusionLitModule(
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        t_max_cosine=args.t_max_cosine,
        warmup_epochs=args.warmup_epochs,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.patience, min_delta=1e-5
    )
    lr_cb  = LearningRateMonitor(logging_interval="epoch")
    pb_cb  = TQDMProgressBar(refresh_rate=10)
    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps    = -1
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
    # Timeout increased to 1800s: setup() loads the pre-computed GNN cache which is
    # fast, but tokenization of 1416 samples can still take 30-60 seconds.
    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=1800))

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
        deterministic="warn",
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)


if __name__ == "__main__":
    main()
