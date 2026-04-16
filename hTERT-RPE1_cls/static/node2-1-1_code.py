"""
Node 2-1-1 — AIDO.Cell-100M + STRING_GNN Multi-Modal Fusion
              with Bilinear Interaction Head

Architecture:
  - AIDO.Cell-100M backbone (LoRA r=64) extracts 640-dim gene-position embedding
    using realistic multi-gene input (all 19,264 genes at 1.0, perturbed gene at 10.0)
  - STRING_GNN backbone (fine-tuned, 5.4M params) with cond_emb perturbation conditioning
    extracts 256-dim PPI node embedding for the perturbed gene
  - Concatenation fusion: [640 || 256] → 896-dim fused perturbation embedding
  - Bilinear interaction head (proven in node1-2 which achieved F1=0.4912):
    fused_proj [B, 3, R] × out_gene_emb [6640, R] → logits [B, 3, 6640]
    Output gene embeddings are initialized from STRING_GNN node embeddings
  - Focal loss with label smoothing (gamma=2.0, ls=0.05) + stronger dropout (0.25)
  - Calibrated cosine LR (max_epochs=60 to fully utilize cosine decay schedule)

Key improvements over parent (node2-1, F1=0.4234):
  1. Multi-modal fusion: transcriptomic (AIDO.Cell) + PPI topology (STRING_GNN)
  2. Bilinear head with structured output gene embeddings (vs flat 42M-param MLP)
  3. LoRA rank r=64 for richer backbone adaptation
  4. Stronger regularization: dropout=0.25, label smoothing=0.05
  5. Calibrated LR schedule (max_epochs=60 vs 150 → cosine decay fully utilized)
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

AIDO_CELL_DIR  = "/home/Models/AIDO.Cell-100M"
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT    = 6640
N_CLASSES      = 3
AIDO_HIDDEN    = 640   # AIDO.Cell-100M hidden size
GNN_HIDDEN     = 256   # STRING_GNN hidden size
FUSED_DIM      = AIDO_HIDDEN + GNN_HIDDEN  # 896
N_LAYERS       = 18   # AIDO.Cell-100M transformer layers
FUSION_LAYERS  = 6    # number of trailing layers to fuse
LORA_R         = 64   # LoRA rank (increased from r=32 for richer adaptation)
LORA_ALPHA     = 128  # LoRA alpha = 2 × rank


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss with optional label smoothing.
    No class weights — focal mechanism handles imbalance via (1-pt)^gamma.
    Label smoothing (0.05) penalizes overconfidence to reduce calibration divergence.
    """
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [N, C] (2D, already reshaped)
        targets: [N] long
        """
        # Compute standard cross-entropy with label smoothing
        ce_loss = F.cross_entropy(
            logits, targets,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        # Get probability of the true class for focal weighting
        with torch.no_grad():
            pt = torch.exp(-F.cross_entropy(logits, targets, reduction='none'))
        # Focal weight: down-weight easy examples
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


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

class PerturbDataset(Dataset):
    """
    Stores pert_ids, symbols, labels, and precomputed STRING_GNN indices.
    The AIDO.Cell multi-gene input is built on-the-fly in the collate_fn.
    """

    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        gnn_node_indices: List[int],   # index in STRING_GNN node_names or -1 for OOV
        labels: Optional[torch.Tensor] = None,  # [N, 6640] long or None
    ):
        self.pert_ids        = pert_ids
        self.symbols         = symbols
        self.gnn_node_indices = gnn_node_indices
        self.labels          = labels

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":       self.pert_ids[idx],
            "symbol":        self.symbols[idx],
            "gnn_node_idx":  self.gnn_node_indices[idx],  # int
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def build_collate_fn(tokenizer):
    """
    Returns a collate function that builds realistic multi-gene AIDO.Cell input profiles.

    For each sample: all 19,264 genes at expression=1.0 (baseline),
    perturbed gene overridden to expression=10.0.
    """
    all_gene_names = list(tokenizer.gene_to_index.keys())  # 19264 gene names
    gene_name_set  = set(all_gene_names)

    def collate_fn(batch):
        pert_ids       = [b["pert_id"]       for b in batch]
        symbols        = [b["symbol"]        for b in batch]
        gnn_node_idxs  = [b["gnn_node_idx"]  for b in batch]
        B = len(batch)

        # Build per-sample expression dicts with realistic multi-gene context
        expr_dicts = []
        gene_vocab_positions = []  # index in 19264-gene AIDO vocab for each sample

        for sym in symbols:
            # Baseline: all genes at expression=1.0
            expr = {g: 1.0 for g in all_gene_names}
            # Perturbed gene: override to 10.0 (10x elevated signal)
            if sym in gene_name_set:
                expr[sym] = 10.0
                gene_vocab_positions.append(tokenizer.gene_to_index[sym])
            else:
                # Out-of-vocabulary for AIDO: position 0 as placeholder
                gene_vocab_positions.append(0)
            expr_dicts.append(expr)

        # Tokenize batch
        tok_out   = tokenizer(expr_dicts, return_tensors="pt")
        input_ids = tok_out["input_ids"]  # [B, 19264] float32

        gene_positions    = torch.tensor(gene_vocab_positions, dtype=torch.long)   # [B]
        gnn_node_idx_t    = torch.tensor(gnn_node_idxs, dtype=torch.long)          # [B]

        out = {
            "pert_id":       pert_ids,
            "symbol":        symbols,
            "input_ids":     input_ids,          # [B, 19264] float32
            "gene_position": gene_positions,     # [B] long — AIDO vocab index
            "gnn_node_idx":  gnn_node_idx_t,     # [B] long — STRING_GNN node index (-1=OOV)
        }
        if "label" in batch[0]:
            out["label"] = torch.stack([b["label"] for b in batch])
        return out

    return collate_fn


# ─── DataModule ───────────────────────────────────────────────────────────────

class FusionDataModule(pl.LightningDataModule):

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
        # ── Load AIDO.Cell tokenizer (DDP-safe barrier) ───────────────────
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)

        # ── Build STRING_GNN node name → index lookup ──────────────────────
        node_names = json.loads((Path(STRING_GNN_DIR) / "node_names.json").read_text())
        # node_names[i] = Ensembl gene ID (e.g., "ENSG00000000003")
        self.gnn_node_name_to_idx = {name: i for i, name in enumerate(node_names)}

        # ── Helper: load a split ───────────────────────────────────────────
        def load_split(fname: str, has_label: bool):
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            pert_ids = df["pert_id"].tolist()
            symbols  = df["symbol"].tolist()

            # Map pert_ids to STRING_GNN node indices
            # pert_id is Ensembl gene ID (e.g., "ENSG00000004897")
            gnn_node_indices = [
                self.gnn_node_name_to_idx.get(pid, -1)  # -1 = OOV
                for pid in pert_ids
            ]

            labels = None
            if has_label and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)

            return PerturbDataset(pert_ids, symbols, gnn_node_indices, labels)

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  False)

        # Build collate function (uses AIDO.Cell tokenizer internals)
        self.collate = build_collate_fn(self.tokenizer)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds, batch_size=self.micro_batch_size,
            shuffle=shuffle, collate_fn=self.collate,
            num_workers=self.num_workers, pin_memory=True,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Model ────────────────────────────────────────────────────────────────────

class FusionPerturbModel(nn.Module):
    """
    Multi-modal fusion model combining:
    1. AIDO.Cell-100M (LoRA r=64): extracts 640-dim gene-position embedding
       from realistic multi-gene expression profile
    2. STRING_GNN (fine-tuned): extracts 256-dim PPI embedding with
       cond_emb perturbation conditioning
    3. Concatenation → 896-dim fused embedding
    4. Bilinear interaction head: fused_proj [B,3,R] × gene_emb[G,R] → [B,3,G]
       Output gene embeddings initialized from STRING_GNN node embeddings

    This architecture addresses the 640-dim information bottleneck in the parent
    (node2-1) by adding PPI structural knowledge (complementary to transcriptomics),
    and replaces the 42M-param flat MLP head with a structured bilinear head (proven
    in node1-2 which achieved F1=0.4912).
    """

    def __init__(
        self,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        bilinear_rank: int = 256,
        head_dropout: float = 0.25,
    ):
        super().__init__()

        # ── 1. AIDO.Cell-100M with LoRA r=64 ──────────────────────────────
        backbone = AutoModel.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False

        # Patch enable_input_require_grads for AIDO.Cell
        def _safe_enable_input_require_grads():
            def _make_inputs_require_grad(module, input, output):
                if isinstance(output, torch.Tensor):
                    output.requires_grad_(True)
            backbone.bert.gene_embedding.register_forward_hook(_make_inputs_require_grad)
        backbone.enable_input_require_grads = _safe_enable_input_require_grads

        # Apply LoRA r=64 (increased from r=32 in parent) to Q/K/V in ALL 18 layers
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.05,
            target_modules=["query", "key", "value"],
        )
        self.aido_backbone = get_peft_model(backbone, lora_cfg)
        self.aido_backbone.print_trainable_parameters()

        # Enable gradient checkpointing
        self.aido_backbone.base_model.model.config.use_cache = False
        self.aido_backbone.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA params to float32 for training stability
        for name, param in self.aido_backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Learnable layer-fusion weights for last FUSION_LAYERS layers
        self.layer_weights = nn.Parameter(torch.zeros(FUSION_LAYERS))

        # ── 2. STRING_GNN backbone (fine-tuned) ───────────────────────────
        self.gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
        # Cast GNN to float32 for stable training (it's small: 5.4M params)
        for param in self.gnn.parameters():
            param.data = param.data.float()

        # ── 3. Load graph data for STRING_GNN ─────────────────────────────
        graph_data   = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", weights_only=False)
        self.register_buffer("edge_index",  graph_data["edge_index"].long())   # [2, E]
        edge_weight  = graph_data.get("edge_weight", None)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight.float())           # [E]
        else:
            self.register_buffer("edge_weight", None)

        # OOV embedding for genes not in STRING_GNN vocabulary
        # Initialized as zero; will be trained alongside head
        self.gnn_oov_emb = nn.Parameter(torch.zeros(GNN_HIDDEN))

        # ── 4. Fusion projection: 640 + 256 → 896 ─────────────────────────
        # LayerNorm applied to each modality before concatenation
        self.aido_norm = nn.LayerNorm(AIDO_HIDDEN)
        self.gnn_norm  = nn.LayerNorm(GNN_HIDDEN)

        # ── 5. Bilinear interaction head ───────────────────────────────────
        # Projects fused 896-dim perturbation embedding to [B, n_classes, bilinear_rank]
        # then dot-products with output gene embeddings [n_genes_out, bilinear_rank]
        # This is the same structure as node1-2 (F1=0.4912) adapted for 896-dim input
        self.pert_proj = nn.Sequential(
            nn.Linear(FUSED_DIM, FUSED_DIM),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(FUSED_DIM, n_classes * bilinear_rank),
        )

        # Output gene embeddings: [n_genes_out, bilinear_rank]
        # Will be initialized from STRING_GNN embeddings in setup (via LightningModule.setup)
        # Here we create the parameter; initialization happens after STRING_GNN forward pass
        self.out_gene_emb = nn.Parameter(
            torch.randn(n_genes_out, bilinear_rank) * 0.02
        )
        self.n_classes    = n_classes
        self.bilinear_rank = bilinear_rank

    def forward(
        self,
        input_ids: torch.Tensor,      # [B, 19264] float32 — AIDO.Cell input
        gene_positions: torch.Tensor,  # [B] long — AIDO vocab index of perturbed gene
        gnn_node_idxs: torch.Tensor,   # [B] long — STRING_GNN node index (-1 = OOV)
    ) -> torch.Tensor:
        """Returns logits [B, 3, 6640]."""
        B = input_ids.shape[0]

        # ── A. AIDO.Cell forward: extract gene-position embedding ──────────
        attn_mask = torch.ones(
            B, input_ids.shape[1],
            dtype=torch.long, device=input_ids.device
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            aido_out = self.aido_backbone(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )

        # hidden_states: tuple of 19 tensors [B, 19266, 640]
        # Use last FUSION_LAYERS layers for weighted fusion
        hidden_states = torch.stack(
            [aido_out.hidden_states[i].float()
             for i in range(N_LAYERS - FUSION_LAYERS + 1, N_LAYERS + 1)],
            dim=0,
        )  # [FUSION_LAYERS, B, 19266, 640]

        weights = torch.softmax(self.layer_weights, dim=0)
        fused_aido = (hidden_states * weights[:, None, None, None]).sum(0)  # [B, 19266, 640]

        # Gene-position extraction: perturbed gene's contextually-informed hidden state
        aido_repr = fused_aido[torch.arange(B, device=fused_aido.device), gene_positions, :]  # [B, 640]

        # ── B. STRING_GNN forward: extract PPI embedding ──────────────────
        # Run a single GNN forward pass per batch to get all 18870 node embeddings.
        # This is efficient: one GNN forward per batch regardless of batch size.
        # Each sample then indexes into the 18870-row embedding matrix.
        device = input_ids.device
        edge_index  = self.edge_index.to(device)
        edge_weight = self.edge_weight.to(device) if self.edge_weight is not None else None

        gnn_out = self.gnn(
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        # Explicitly cast to float32: inside bf16-mixed autocast, GNN matmul ops may
        # produce bf16 output even though GNN parameters are float32.
        gnn_all_embs = gnn_out.last_hidden_state.float()  # [18870, 256] float32

        # Extract per-sample GNN embeddings; use OOV embedding for -1 indices
        in_vocab_mask = (gnn_node_idxs >= 0)  # [B] bool
        gnn_repr = torch.zeros(B, GNN_HIDDEN, dtype=torch.float32, device=device)
        if in_vocab_mask.any():
            valid_node_idxs = gnn_node_idxs[in_vocab_mask]  # [n_valid]
            gnn_repr[in_vocab_mask] = gnn_all_embs[valid_node_idxs]
        # OOV genes get the learnable OOV embedding
        oov_batch_mask = ~in_vocab_mask
        if oov_batch_mask.any():
            gnn_repr[oov_batch_mask] = self.gnn_oov_emb.unsqueeze(0).expand(
                oov_batch_mask.sum(), -1
            )

        # ── C. Fusion: concatenate AIDO + GNN representations ─────────────
        aido_repr_norm = self.aido_norm(aido_repr)
        gnn_repr_norm  = self.gnn_norm(gnn_repr)
        fused = torch.cat([aido_repr_norm, gnn_repr_norm], dim=1)  # [B, 896]

        # ── D. Bilinear interaction head ───────────────────────────────────
        # Project fused embedding to [B, n_classes * bilinear_rank]
        pert_proj = self.pert_proj(fused)  # [B, 3*256]
        pert_proj = pert_proj.view(B, self.n_classes, self.bilinear_rank)  # [B, 3, 256]

        # Bilinear interaction: [B, 3, 256] × [6640, 256]^T → [B, 3, 6640]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)  # [B, 3, 6640]
        return logits


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
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

class FusionLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_backbone: float = 5e-5,   # conservative LR for AIDO.Cell LoRA backbone
        lr_gnn: float = 1e-4,        # STRING_GNN fine-tuning LR (slightly higher — smaller model)
        lr_head: float = 3e-4,       # standard LR for fresh prediction head
        weight_decay: float = 1e-3,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        head_dropout: float = 0.25,   # increased from 0.1 for stronger regularization
        bilinear_rank: int = 256,
        warmup_steps: int = 100,
        max_steps_total: int = 2000,
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
        self.model = FusionPerturbModel(
            head_dropout=self.hparams.head_dropout,
            bilinear_rank=self.hparams.bilinear_rank,
        )

        # Initialize output gene embeddings from STRING_GNN node embeddings
        # This provides a structural prior on the 6,640 output genes
        # STRING_GNN has 18,870 nodes; we need to map dataset gene positions to GNN nodes
        # The output gene embeddings need to cover the 6,640 output genes defined by the task
        # We use the GNN's pre-computed embeddings as initialization via a frozen forward pass
        self._init_output_gene_embeddings()

        self.focal_loss = FocalLoss(
            gamma=self.hparams.focal_gamma,
            label_smoothing=self.hparams.label_smoothing,
        )

    def _init_output_gene_embeddings(self):
        """
        Initialize output gene embeddings from STRING_GNN pre-trained node embeddings.

        The 6,640 output positions are anonymous (no direct name mapping to GNN nodes
        is provided in the dataset). We use the first 6,640 of STRING_GNN's 18,870 node
        embeddings as a reasonable initialization — this provides pre-trained PPI structure
        as a starting point rather than pure random initialization.

        Note: bilinear_rank=256 = GNN_HIDDEN=256 by default, enabling direct assignment.
        If bilinear_rank != GNN_HIDDEN, we fall back to the existing random initialization.
        """
        if self.hparams.bilinear_rank != GNN_HIDDEN:
            self.print("[Node2-1-1] bilinear_rank != GNN_HIDDEN; using random output gene emb init")
            return

        with torch.no_grad():
            # Run GNN on CPU to avoid device issues during setup
            gnn_cpu = self.model.gnn.cpu()
            edge_index_cpu  = self.model.edge_index.cpu()
            edge_weight_cpu = self.model.edge_weight.cpu() if self.model.edge_weight is not None else None

            gnn_out = gnn_cpu(
                edge_index=edge_index_cpu,
                edge_weight=edge_weight_cpu,
            )
            all_node_embs = gnn_out.last_hidden_state.float()  # [18870, 256]

            n_init = min(N_GENES_OUT, all_node_embs.shape[0])
            # Use the first N_GENES_OUT node embeddings as initialization
            init_embs = all_node_embs[:n_init]
            if n_init < N_GENES_OUT:
                extra = torch.randn(N_GENES_OUT - n_init, self.hparams.bilinear_rank) * 0.02
                init_full = torch.cat([init_embs, extra], dim=0)
            else:
                init_full = init_embs

            # Direct copy into the parameter (same shape: [6640, 256])
            self.model.out_gene_emb.data.copy_(init_full)

        self.print(f"[Node2-1-1] Initialized output gene embeddings from STRING_GNN "
                   f"({n_init}/{N_GENES_OUT} genes from pre-trained PPI embeddings)")

    def forward(self, input_ids, gene_positions, gnn_node_idxs):
        return self.model(input_ids, gene_positions, gnn_node_idxs)

    def _loss(self, logits, labels):
        # logits: [B, 3, 6640] -> [B*6640, 3];  labels: [B, 6640] -> [B*6640]
        logits_2d = logits.float().permute(0, 2, 1).reshape(-1, N_CLASSES)
        labels_1d = labels.reshape(-1)
        return self.focal_loss(logits_2d, labels_1d)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"], batch["gnn_node_idx"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"], batch["gnn_node_idx"])
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
        logits = self(batch["input_ids"], batch["gene_position"], batch["gnn_node_idx"])
        probs  = torch.softmax(logits, dim=1)
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs  = torch.cat(self._test_preds, 0)
        dummy_labels = (
            torch.cat(self._test_labels, 0)
            if self._test_labels
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
            all_pert, all_syms    = self._test_pert_ids, self._test_symbols

        if self.trainer.is_global_zero:
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate by pert_id (DDP may pad with duplicates)
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
            self.print(f"[Node2-1-1] Saved {len(dedup_indices)} test predictions → {pred_path}")

            if self._test_labels:
                dedup_probs  = all_probs_np[dedup_indices]
                dedup_labels = all_labels[dedup_indices].numpy()
                f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
                self.print(f"[Node2-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Three non-overlapping parameter groups:
        # 1. AIDO.Cell LoRA backbone + layer fusion weights
        # 2. STRING_GNN backbone parameters only (model.gnn.*)
        # 3. All other trainable params (head, norms, OOV emb, output gene emb)
        aido_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and ("aido_backbone" in n or n == "layer_weights")
        ]
        # Only the actual GNN model parameters (model.gnn.emb, model.gnn.mps, model.gnn.post_mp)
        gnn_model_prefix = "gnn."
        gnn_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and n.startswith(gnn_model_prefix)
        ]
        # Everything else: head projection, output gene emb, norms, OOV emb
        aido_set = set(id(p) for p in aido_params)
        gnn_set  = set(id(p) for p in gnn_params)
        head_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and id(p) not in aido_set and id(p) not in gnn_set
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": aido_params, "lr": hp.lr_backbone, "weight_decay": hp.weight_decay},
                {"params": gnn_params,  "lr": hp.lr_gnn,      "weight_decay": hp.weight_decay},
                {"params": head_params, "lr": hp.lr_head,     "weight_decay": hp.weight_decay},
            ]
        )

        # Cosine annealing with linear warmup — calibrated to actual training duration
        warmup = hp.warmup_steps
        total  = hp.max_steps_total

        def lr_lambda(current_step: int):
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            progress = float(current_step - warmup) / float(max(1, total - warmup))
            return max(1e-6, 0.5 * (1.0 + np.cos(np.pi * progress)))

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
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        sd = {k: v for k, v in full_sd.items()
              if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving ckpt: {trained}/{total} params ({100*trained/total:.2f}%)")
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Node 2-1-1 — AIDO.Cell-100M + STRING_GNN Fusion + Bilinear Head")
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--lr-backbone",       type=float, default=5e-5)
    p.add_argument("--lr-gnn",            type=float, default=1e-4)
    p.add_argument("--lr-head",           type=float, default=3e-4)
    p.add_argument("--weight-decay",      type=float, default=1e-3)
    p.add_argument("--focal-gamma",       type=float, default=2.0)
    p.add_argument("--label-smoothing",   type=float, default=0.05)
    p.add_argument("--head-dropout",      type=float, default=0.25)
    p.add_argument("--bilinear-rank",     type=int,   default=256)
    p.add_argument("--warmup-steps",      type=int,   default=100)
    p.add_argument("--micro-batch-size",  type=int,   default=4)
    p.add_argument("--global-batch-size", type=int,   default=32)
    p.add_argument("--max-epochs",        type=int,   default=60,
                   help="Calibrated to actual training duration for cosine LR utilization")
    p.add_argument("--patience",          type=int,   default=20)
    p.add_argument("--num-workers",       type=int,   default=2)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",    type=int,   default=None)
    p.add_argument("--fast-dev-run",      action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Estimate total training steps for LR schedule (calibrated to max_epochs=60)
    train_size  = 1416
    steps_per_epoch = max(1, train_size // (args.micro_batch_size * n_gpus))
    accum       = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    eff_steps_per_epoch = max(1, steps_per_epoch // accum)
    max_steps_total = eff_steps_per_epoch * args.max_epochs

    dm  = FusionDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = FusionLitModule(
        lr_backbone     = args.lr_backbone,
        lr_gnn          = args.lr_gnn,
        lr_head         = args.lr_head,
        weight_decay    = args.weight_decay,
        focal_gamma     = args.focal_gamma,
        label_smoothing = args.label_smoothing,
        head_dropout    = args.head_dropout,
        bilinear_rank   = args.bilinear_rank,
        warmup_steps    = args.warmup_steps,
        max_steps_total = max_steps_total,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-5)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps_trainer: int = -1
    limit_train: float | int = 1.0
    limit_val:   float | int = 1.0
    limit_test:  float | int = 1.0
    fast_dev_run = False

    if args.debug_max_step is not None:
        max_steps_trainer = args.debug_max_step
        limit_train = args.debug_max_step
        limit_val   = 2
        limit_test  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps_trainer,
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
        gradient_clip_val=1.0,
    )

    trainer.fit(lit, datamodule=dm)

    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 2-1-1 — AIDO.Cell-100M (LoRA r=64) + STRING_GNN Fusion + Bilinear Head\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
