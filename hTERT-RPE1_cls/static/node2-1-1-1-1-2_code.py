"""
Node 2-1-1-1-1-2 — Partial STRING_GNN Fine-tuning (mps.6+7+post_mp) + rank=512 + Class-Weighted Focal Loss

Architecture directly implements the tree-best strategy from node2-1-3 (F1=0.5047):
  - Partial STRING_GNN backbone fine-tuning:
      * mps.0-5: frozen (pre-computed once in DataModule.setup() → stored as persistent buffer)
      * mps.6, mps.7, post_mp: trainable (~198K params)
      * In each forward pass: apply trainable tail over all 18870 nodes, extract batch rows
  - rank=512 6-layer deep residual bilinear head (~16.9M trainable params):
      * Larger interaction space than parent's rank=256 (3.4M vs 1.7M for out_gene_emb)
      * 6 layers proven optimal (8 layers hurts: node2-1-1-1-1-1-1 F1=0.4705 < node1-2 F1=0.4912)
  - Class-weighted focal loss (gamma=2.0, weights=[down=2.0, neutral=0.5, up=4.0]):
      * Proven effective: node2-1-2 (F1=0.5011) and node2-1-3 (F1=0.5047, tree best)
      * Mild neutral downweight (0.5) reduces dominance of 88.9% neutral class
  - Two-group AdamW: backbone_lr=5e-5, head_lr=5e-4
  - Flat cosine LR: total_steps=6600 (barely decays during training, proven in node2-1-3)
  - Patience=50, max_epochs=300

Distinct from sibling node2-1-1-1-1-1 (F1=0.4780) in:
  - Backbone: trainable mps.6+7+post_mp vs fully frozen
  - Layers: 6 vs 8 (8 layers proven worse)
  - Loss: class weights [2.0, 0.5, 4.0] vs no class weights + label smoothing
  - Dropout: 0.2 vs 0.15
  - LR: two-group (5e-5/5e-4) vs single (5e-4)

Expected performance: F1 >= 0.500 based on tree best evidence (node2-1-3: 0.5047)

Key memory references:
  - node2-1-3 (F1=0.5047, tree best): partial FT + rank=512 + class_weights=[2.0,0.5,4.0]
  - node2-1-2 (F1=0.5011): frozen + rank=256 + class_weights=[1.5,0.8,2.5]
  - node2-1-1-1-1-1-1 (F1=0.4705): 8 layers with rank=256 is WORSE than 6 layers
  - node2-1-1-1-1-1 (F1=0.4780): sibling, frozen + 8 layers + label_smoothing; marginal gain
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Required for deterministic=True with CUDA >= 10.2

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
from lightning.pytorch.strategies import DDPStrategy
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# ─── Constants ────────────────────────────────────────────────────────────────

N_GENES_OUT = 6640
N_CLASSES = 3
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")

# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_logits_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Exact per-gene macro F1 matching calc_metric.py logic.

    Args:
        pred_logits_np: [N, 3, G] float (logits or probabilities)
        labels_np:      [N, G]    int   (class indices 0/1/2)

    Returns:
        Mean per-gene F1 score (float).
    """
    pred_classes = pred_logits_np.argmax(axis=1)  # [N, G]
    n_genes = labels_np.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = labels_np[:, g]
        yh = pred_classes[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset.

    Stores node indices (integers) for STRING_GNN lookup — not precomputed embeddings.
    The model computes fresh embeddings at each forward pass via the trainable GNN tail.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        node_name_to_idx: Dict[str, int],
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.has_labels = has_labels

        n_samples = len(df)
        node_idxs = []
        in_vocab = []

        for pert_id in self.pert_ids:
            if pert_id in node_name_to_idx:
                node_idxs.append(node_name_to_idx[pert_id])
                in_vocab.append(True)
            else:
                node_idxs.append(0)  # placeholder (in_vocab=False signals this is OOV)
                in_vocab.append(False)

        self.node_idxs = torch.tensor(node_idxs, dtype=torch.long)  # [N]
        self.in_vocab = torch.tensor(in_vocab, dtype=torch.bool)    # [N]

        if has_labels and "label" in df.columns:
            rows = []
            for lbl_str in df["label"]:
                rows.append([x + 1 for x in json.loads(lbl_str)])
            self.labels = torch.tensor(rows, dtype=torch.long)  # [N, G]
        else:
            self.has_labels = False

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int):
        item = {
            "pert_id":  self.pert_ids[idx],
            "symbol":   self.symbols[idx],
            "node_idx": self.node_idxs[idx],  # int index into STRING_GNN nodes
            "in_vocab": self.in_vocab[idx],    # bool
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule.

    Precomputes frozen STRING_GNN intermediate embeddings (output after mps.5) once.
    These are stored as `mid_embs` [18870, 256] for use as input to the trainable GNN tail.
    The model runs mps.6 → mps.7 → post_mp in each forward pass to get fresh embeddings.
    """

    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        # Guard against double initialization
        if hasattr(self, "train_ds"):
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("[DataModule] Loading STRING_GNN and computing frozen intermediate embeddings...")
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", weights_only=False)

        model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        edge_index = graph["edge_index"].to(device)
        edge_weight = graph["edge_weight"].to(device) if graph.get("edge_weight") is not None else None

        # Run through all 8 GCN layers with output_hidden_states=True
        # hidden_states[6] = output after mps.5 (input to mps.6) = mid_embs
        with torch.no_grad():
            outputs = model(
                edge_index=edge_index,
                edge_weight=edge_weight,
                output_hidden_states=True,
            )

        # mid_embs: frozen output after mps.5 (index 6 = initial + 6 layers = after mps.5)
        # hidden_states: tuple of 9 [initial, after mps.0, ..., after mps.7]
        # hidden_states[0] = initial emb.weight
        # hidden_states[1] = after mps.0
        # ...
        # hidden_states[6] = after mps.5  <-- this is mid_embs
        # hidden_states[7] = after mps.6
        # hidden_states[8] = after mps.7
        self.mid_embs = outputs.hidden_states[6].float().cpu().numpy()  # [18870, 256]

        # Store graph topology for model's GNN forward passes
        self.edge_index = graph["edge_index"]  # [2, E] cpu tensor
        self.edge_weight = graph["edge_weight"] if graph.get("edge_weight") is not None else None

        self.node_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(node_names)}
        self.n_gnn_nodes = len(node_names)

        del model
        torch.cuda.empty_cache()

        print(f"[DataModule] Mid-embeddings shape (after mps.5): {self.mid_embs.shape}")
        print(f"[DataModule] Graph: {self.n_gnn_nodes} nodes, edge_index: {self.edge_index.shape}")

        # Load all splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        n_train_cov = sum(p in self.node_name_to_idx for p in dfs["train"]["pert_id"])
        n_val_cov = sum(p in self.node_name_to_idx for p in dfs["val"]["pert_id"])
        print(f"[DataModule] Coverage: {n_train_cov}/{len(dfs['train'])} train genes, "
              f"{n_val_cov}/{len(dfs['val'])} val genes in STRING_GNN")

        self.train_ds = PerturbationDataset(dfs["train"], self.node_name_to_idx, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   self.node_name_to_idx, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  self.node_name_to_idx, True)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )


# ─── Model Components ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual MLP block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout + skip."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GNNBilinearHead(nn.Module):
    """Rank-512 deep bilinear prediction head (6 residual layers).

    Same proven design as node1-2 / node2-1-3, but with rank=512 for richer
    bilinear interaction space between perturbation embedding and output genes.

    Left side: GNN embedding -> Deep MLP -> [B, 3, rank]
    Right side: randomly initialized learnable output gene embeddings [n_genes_out, rank]
    Interaction: einsum("bcr,gr->bcg") -> logits [B, 3, n_genes_out]
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        dropout: float = 0.2,
        n_residual_layers: int = 6,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_genes_out = n_genes_out
        self.rank = rank

        # Input normalization
        self.input_norm = nn.LayerNorm(gnn_dim)

        # Projection: gnn_dim -> hidden_dim
        self.proj_in = nn.Linear(gnn_dim, hidden_dim)

        # Deep residual MLP
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, expand=4, dropout=dropout)
             for _ in range(n_residual_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear projection: hidden_dim -> n_classes * rank
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * rank)

        # Output gene embeddings: [n_genes_out, rank] — random init (proven in node2-1-3)
        self.out_gene_emb = nn.Parameter(torch.empty(n_genes_out, rank))

        # Head dropout
        self.head_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)
        nn.init.normal_(self.out_gene_emb, std=0.02)  # random init (proven in node2-1-3)

    def forward(self, gnn_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gnn_emb: [B, gnn_dim] - STRING_GNN embeddings after partial fine-tuning
        Returns:
            logits: [B, 3, n_genes_out]
        """
        B = gnn_emb.shape[0]

        x = self.input_norm(gnn_emb)
        x = self.proj_in(x)   # [B, hidden_dim]

        for blk in self.res_blocks:
            x = blk(x)
        x = self.norm_out(x)   # [B, hidden_dim]

        x = self.head_dropout(x)
        pert_proj = self.proj_bilinear(x)                          # [B, n_classes * rank]
        pert_proj = pert_proj.view(B, self.n_classes, self.rank)   # [B, 3, rank]

        # Bilinear interaction: [B, 3, rank] x [n_genes_out, rank]^T -> [B, 3, n_genes_out]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)
        return logits


# ─── Class-Weighted Focal Loss ─────────────────────────────────────────────────

def focal_loss_with_class_weights(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    class_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """Focal loss with class weights.

    Args:
        logits: [B, C, G] logits
        labels: [B, G] long class indices (0/1/2)
        gamma:  focal exponent (2.0 proven effective)
        class_weights: [C] per-class weight multipliers
            [2.0, 0.5, 4.0] for [down, neutral, up]:
            - Class 0 (down, 8.1% freq): 2.0x → moderate upweighting
            - Class 1 (neutral, 88.9% freq): 0.5x → downweight dominant class
            - Class 2 (up, 3.0% freq): 4.0x → strongest upweighting for rarest class
    Returns:
        Scalar loss
    """
    B, C, G = logits.shape
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
    labels_flat = labels.reshape(-1)                        # [B*G]

    # Build class weight tensor on the right device
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(
            class_weights, dtype=logits_flat.dtype, device=logits_flat.device
        )

    ce_loss = F.cross_entropy(
        logits_flat, labels_flat,
        weight=weight_tensor,
        reduction="none"
    )  # [B*G]

    with torch.no_grad():
        probs = F.softmax(logits_flat, dim=1)
        pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt).pow(gamma)

    return (focal_weight * ce_loss).mean()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(
    local_preds: torch.Tensor,
    local_labels: torch.Tensor,
    device: torch.device,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather variable-length tensors from all DDP ranks with padding."""
    local_size = torch.tensor([local_preds.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = int(max(s.item() for s in all_sizes))

    pad = max_size - local_preds.shape[0]
    p = local_preds.to(device)
    l = local_labels.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], dim=0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], dim=0)

    g_preds  = [torch.zeros_like(p) for _ in range(world_size)]
    g_labels = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(g_preds,  p)
    dist.all_gather(g_labels, l)

    real_preds  = torch.cat([g_preds[i][:all_sizes[i].item()].cpu()  for i in range(world_size)], 0)
    real_labels = torch.cat([g_labels[i][:all_sizes[i].item()].cpu() for i in range(world_size)], 0)
    return real_preds, real_labels


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbationLitModule(pl.LightningModule):
    """LightningModule for gene-perturbation DEG prediction.

    Partial STRING_GNN fine-tuning (mps.6, mps.7, post_mp trainable) + rank-512 bilinear head.
    Backbone parameters run over the full 18870-node graph at each forward pass.
    Class-weighted focal loss with two-group AdamW optimization.
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_residual_layers: int = 6,
        dropout: float = 0.2,
        backbone_lr: float = 5e-5,
        head_lr: float = 5e-4,
        weight_decay: float = 1e-3,
        focal_gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,
        warmup_steps: int = 100,
        total_steps: int = 6600,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None):
        if hasattr(self, "head"):
            return  # already initialized

        hp = self.hparams
        dm = self.trainer.datamodule

        # ── Step 1: Load trainable GNN tail (mps.6, mps.7, post_mp) ────────────
        print("[Setup] Loading STRING_GNN for trainable tail extraction...")
        model_full = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        model_full.eval()

        # Extract the trainable sub-modules (last 2 GCN layers + output projection)
        self.gnn_mps6    = model_full.mps[6]   # 7th GCN layer (trainable)
        self.gnn_mps7    = model_full.mps[7]   # 8th GCN layer (trainable)
        self.gnn_post_mp = model_full.post_mp  # output projection Linear(256, 256) (trainable)

        # Delete the rest to free memory (we don't need emb, mps.0-5 — they're in mid_embs)
        del model_full
        torch.cuda.empty_cache()

        # Make trainable tail parameters float32 for stable optimization
        for p in self.gnn_mps6.parameters():
            p.requires_grad = True
            p.data = p.data.float()
        for p in self.gnn_mps7.parameters():
            p.requires_grad = True
            p.data = p.data.float()
        for p in self.gnn_post_mp.parameters():
            p.requires_grad = True
            p.data = p.data.float()

        # ── Step 2: Register frozen intermediate embeddings as non-persistent buffers ──
        # persistent=False: not saved in checkpoint (recomputed from STRING_GNN if needed)
        mid_embs_tensor = torch.from_numpy(dm.mid_embs).float()  # [18870, 256]
        self.register_buffer("mid_embs_buffer", mid_embs_tensor, persistent=False)

        # Register graph topology buffers (non-persistent: derived from STRING_GNN graph file)
        self.register_buffer("edge_index_buffer", dm.edge_index.long(), persistent=False)
        if dm.edge_weight is not None:
            self.register_buffer("edge_weight_buffer", dm.edge_weight.float(), persistent=False)
        else:
            self.register_buffer("edge_weight_buffer", None, persistent=False)

        # ── Step 3: Learnable OOV embedding for ~6.4% out-of-vocabulary genes ──
        self.oov_embedding = nn.Parameter(torch.empty(hp.gnn_dim))
        nn.init.normal_(self.oov_embedding, std=0.02)

        # ── Step 4: Build bilinear prediction head ──────────────────────────────
        self.head = GNNBilinearHead(
            gnn_dim=hp.gnn_dim,
            hidden_dim=hp.hidden_dim,
            rank=hp.rank,
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            dropout=hp.dropout,
            n_residual_layers=hp.n_residual_layers,
        )

        # Cast all trainable parameters to float32
        for p in self.head.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # ── Parameter count ─────────────────────────────────────────────────────
        backbone_params = (
            sum(p.numel() for p in self.gnn_mps6.parameters()) +
            sum(p.numel() for p in self.gnn_mps7.parameters()) +
            sum(p.numel() for p in self.gnn_post_mp.parameters())
        )
        oov_params = self.oov_embedding.numel()
        head_params = sum(p.numel() for p in self.head.parameters())
        total_trainable = backbone_params + oov_params + head_params
        print(f"[Setup] Trainable params: backbone_tail={backbone_params:,}, "
              f"oov={oov_params}, head={head_params:,}, total={total_trainable:,}")
        print(f"[Setup] STRING_GNN mps.0-5 + emb are FROZEN (precomputed in DataModule)")
        print(f"[Setup] Trainable tail: mps.6 + mps.7 + post_mp = {backbone_params:,} params")

    def forward(
        self,
        node_idx: torch.Tensor,  # [B] int indices into STRING_GNN nodes
        in_vocab: torch.Tensor,  # [B] bool
    ) -> torch.Tensor:
        """
        Compute logits for a batch of perturbation genes.

        Procedure:
        1. Apply trainable GNN tail (mps.6 → mps.7 → post_mp) over all 18870 nodes
           using the frozen mid_embs_buffer as input.
        2. Extract batch embeddings by node index (or use oov_embedding for OOV genes).
        3. Pass batch embeddings through the bilinear head.

        The full-graph GNN computation in step 1 enables proper message passing.
        Gradients flow through the GNN tail's parameters only (not through mid_embs_buffer).
        """
        # Step 1: Full-graph GNN tail forward (all 18870 nodes)
        # Note: GNNLayer.forward() returns processed x WITHOUT residual.
        # The residual is added here (matching StringGNNModel.forward: x = mp(x,...) + x).
        x = self.mid_embs_buffer.float().detach()  # [18870, 256], no grad (frozen)
        x = self.gnn_mps6(x, self.edge_index_buffer, self.edge_weight_buffer) + x   # [18870, 256]
        x = self.gnn_mps7(x, self.edge_index_buffer, self.edge_weight_buffer) + x   # [18870, 256]
        x = self.gnn_post_mp(x)                                                      # [18870, 256]
        # x now has gradient through gnn_mps6, gnn_mps7, gnn_post_mp

        # Step 2: Extract batch embeddings
        B = node_idx.shape[0]
        batch_emb = torch.zeros(B, self.hparams.gnn_dim, device=x.device, dtype=x.dtype)

        in_vocab_mask = in_vocab  # [B] bool
        if in_vocab_mask.any():
            valid_idxs = node_idx[in_vocab_mask]      # [n_valid] valid node indices
            batch_emb[in_vocab_mask] = x[valid_idxs]  # [n_valid, 256]

        if (~in_vocab_mask).any():
            n_oov = (~in_vocab_mask).sum()
            # Cast oov_embedding to match batch_emb dtype (may be BFloat16 under bf16-mixed precision)
            batch_emb[~in_vocab_mask] = self.oov_embedding.to(batch_emb.dtype).unsqueeze(0).expand(n_oov, -1)

        # Step 3: Bilinear head
        logits = self.head(batch_emb)  # [B, 3, 6640]
        return logits

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return focal_loss_with_class_weights(
            logits, labels,
            gamma=self.hparams.focal_gamma,
            class_weights=self.hparams.class_weights,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_idx"], batch["in_vocab"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["node_idx"], batch["in_vocab"])
        if "label" in batch:
            loss = self._compute_loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())
        return logits

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        local_p = torch.cat(self._val_preds,  dim=0)
        local_l = torch.cat(self._val_labels, dim=0)

        if self.trainer.world_size > 1:
            all_p, all_l = _gather_tensors(local_p, local_l, self.device, self.trainer.world_size)
        else:
            all_p, all_l = local_p, local_l

        f1 = compute_per_gene_f1(all_p.numpy(), all_l.numpy())
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["node_idx"], batch["in_vocab"])
        probs = torch.softmax(logits, dim=1)  # [B, 3, 6640]
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs = torch.cat(self._test_preds, dim=0)
        dummy_labels = torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        if self._test_labels:
            dummy_labels = torch.cat(self._test_labels, dim=0)

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
            all_probs  = local_probs
            all_labels = dummy_labels
            all_pert   = self._test_pert_ids
            all_syms   = self._test_symbols

        # De-duplicate on all ranks (same gathered data on all ranks) for F1 logging
        seen_ids_all: set = set()
        dedup_probs_all: list = []
        dedup_labels_all: list = []
        dedup_pert_all: list = []
        dedup_syms_all: list = []
        all_probs_np = all_probs.numpy()
        for i, (pert_id, symbol) in enumerate(zip(all_pert, all_syms)):
            if pert_id not in seen_ids_all:
                seen_ids_all.add(pert_id)
                dedup_probs_all.append(all_probs_np[i])
                dedup_labels_all.append(all_labels[i].numpy() if isinstance(all_labels[i], torch.Tensor) else all_labels[i])
                dedup_pert_all.append(pert_id)
                dedup_syms_all.append(symbol)

        test_f1 = 0.0
        if dedup_probs_all and dedup_labels_all:
            dedup_probs_np  = np.stack(dedup_probs_all, axis=0)
            dedup_labels_np = np.stack(dedup_labels_all, axis=0)
            if dedup_labels_np.any():
                test_f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                self.print(f"[Node2-1-1-1-1-2] Self-computed test F1 = {test_f1:.4f}")

        # Log test_f1 on all ranks so Lightning can record it in test_results.
        # Value is already computed from globally-gathered data (identical on all ranks).
        self.log("test_f1", test_f1, prog_bar=True, sync_dist=True)

        if self.trainer.is_global_zero:
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for pert_id, symbol, probs in zip(dedup_pert_all, dedup_syms_all, dedup_probs_all):
                    fh.write(f"{pert_id}\t{symbol}\t{json.dumps(probs.tolist())}\n")
            self.print(f"[Node2-1-1-1-1-2] Saved test predictions → {pred_path} ({len(seen_ids_all)} samples)")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Two optimizer groups: backbone tail (low LR) and head (higher LR)
        # This is the proven recipe from node2-1-3 (F1=0.5047):
        #   backbone_lr=5e-5 for GNN layers (slow adaptation to preserve pretrained topology)
        #   head_lr=5e-4 for bilinear head (standard learning rate)
        backbone_params = (
            list(self.gnn_mps6.parameters()) +
            list(self.gnn_mps7.parameters()) +
            list(self.gnn_post_mp.parameters())
        )
        head_params = (
            [self.oov_embedding] +
            list(self.head.parameters())
        )

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": hp.backbone_lr,  "weight_decay": hp.weight_decay},
                {"params": head_params,     "lr": hp.head_lr,      "weight_decay": hp.weight_decay},
            ],
            # Note: lr here is ignored (overridden per group), but needed for AdamW init
            lr=hp.head_lr,
        )

        # Cosine annealing with linear warmup
        # total_steps=6600: INTENTIONALLY LONG to keep LR relatively flat throughout training.
        # This is the proven strategy from node2-1-3 (F1=0.5047): by epoch 82, cosine progress
        # was only ~0.17, meaning LR barely decayed. This flat LR regime allows more sustained
        # learning across all training epochs before patience triggers.
        def lr_lambda(step: int) -> float:
            if step < hp.warmup_steps:
                return float(step) / max(1, hp.warmup_steps)
            progress = (step - hp.warmup_steps) / max(1, hp.total_steps - hp.warmup_steps)
            return max(0.01, 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable params (not large graph buffers) ───────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters (not large non-persistent graph buffers)."""
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        # Only include persistent buffers (exclude mid_embs_buffer, edge_index_buffer, etc.)
        persistent_buffer_keys = set()
        for n, buf in self.named_buffers():
            # Non-persistent buffers are not in state_dict by default; skip them
            # Persistent buffers (e.g., BatchNorm running stats) would be included
            pass  # We rely on the non-persistent registration to exclude large buffers

        trainable_sd = {k: v for k, v in full_sd.items() if k in trainable_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(f"Saving ckpt: {trained}/{total} trainable params ({100*trained/total:.1f}%)")
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        """Load only trainable parameters; ignore buffers (recomputed from DataModule)."""
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 2-1-1-1-1-2 – Partial STRING_GNN Fine-tuning + rank=512 + Class Weights"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--gnn-dim",            type=int,   default=256)
    p.add_argument("--hidden-dim",         type=int,   default=512)
    p.add_argument("--rank",               type=int,   default=512,
                   help="Bilinear rank (512 proven in node2-1-3 tree best)")
    p.add_argument("--n-residual-layers",  type=int,   default=6,
                   help="6 layers proven optimal (8 layers hurts: F1=0.4705 vs 0.4912)")
    p.add_argument("--dropout",            type=float, default=0.2,
                   help="Dropout=0.2 proven in node2-1-3 (0.3 over-regularizes)")
    p.add_argument("--backbone-lr",        type=float, default=5e-5,
                   help="LR for trainable GNN tail (mps.6+7+post_mp)")
    p.add_argument("--head-lr",            type=float, default=5e-4,
                   help="LR for bilinear head and OOV embedding")
    p.add_argument("--weight-decay",       type=float, default=1e-3)
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--class-weights",      type=float, nargs=3,
                   default=[2.0, 0.5, 4.0],
                   help="Class weights [down, neutral, up]: proven in node2-1-3 (F1=0.5047)")
    p.add_argument("--warmup-steps",       type=int,   default=100,
                   help="Linear warmup steps (100 proven in node2-1-3)")
    p.add_argument("--total-steps",        type=int,   default=6600,
                   help="Cosine schedule total steps (6600=flat LR, proven in node2-1-3)")
    p.add_argument("--micro-batch-size",   type=int,   default=16)
    p.add_argument("--global-batch-size",  type=int,   default=64)
    p.add_argument("--max-epochs",         type=int,   default=300)
    p.add_argument("--patience",           type=int,   default=50,
                   help="Early stopping patience (50 proven in node2-1-3)")
    p.add_argument("--num-workers",        type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",     type=int,   default=None)
    p.add_argument("--fast-dev-run",       action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # DataModule: precomputes frozen mid_embs (after mps.5) and stores graph topology
    dm = PerturbationDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    # Gradient accumulation for effective batch size
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    print(f"[Main] n_gpus={n_gpus}, micro_batch={args.micro_batch_size}, "
          f"global_batch={args.global_batch_size}, accum={accum}")
    print(f"[Main] LR schedule: warmup={args.warmup_steps}, total_steps={args.total_steps} "
          f"(flat LR strategy from node2-1-3 tree best)")
    print(f"[Main] class_weights={args.class_weights} (proven in node2-1-3 F1=0.5047)")

    # LightningModule
    lit = PerturbationLitModule(
        gnn_dim=args.gnn_dim,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
        n_residual_layers=args.n_residual_layers,
        dropout=args.dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        class_weights=args.class_weights,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=1, save_last=True,
    )
    es_cb = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.patience,
        min_delta=1e-5,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    pb_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    # Debug / fast-dev-run settings
    max_steps:           int | None  = -1
    limit_train_batches: float | int = 1.0
    limit_val_batches:   float | int = 1.0
    limit_test_batches:  float | int = 1.0
    fast_dev_run = False

    if args.debug_max_step is not None:
        max_steps           = args.debug_max_step
        limit_train_batches = args.debug_max_step
        limit_val_batches   = 2
        limit_test_batches  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    # DDP strategy: always use DDPStrategy (find_unused_parameters=True for oov_embedding
    # which may be unused in batches with all in-vocab genes)
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
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=args.val_check_interval if (args.debug_max_step is None and not args.fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)

    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    test_results = trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            str(test_results[0].get("test_f1", "N/A")) + "\n"
            if test_results else "N/A\n"
        )
        print(f"[Main] Test results: {test_results}")
        print(f"[Main] Score saved to {score_path}")


if __name__ == "__main__":
    main()
