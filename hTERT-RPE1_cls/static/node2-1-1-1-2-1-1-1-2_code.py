"""
Node 2-1-1-1-2-1-1-1-2 — Deeper Backbone (mps.4-7) + Quality-Filtered Top-8 SWA
                         + SGDR T_0=20 + WD=3e-3 + Dropout=0.45

Architecture:
  - STRING_GNN backbone:
      * mps.0-3: frozen (precomputed embeddings through layer 3 stored in dataset)
      * mps.4, mps.5: trainable at very low LR (backbone_lr2=2e-6, AdamW)
      * mps.6, mps.7, post_mp: trainable at backbone_lr=1e-5 (AdamW)
  - Output gene embedding initialization: STRING_GNN mps.0-3 output (sufficient for init)
  - 6-layer deep residual bilinear MLP head (rank=512, hidden=512, expand=4, dropout=0.45)
  - MuonWithAuxAdam optimizer:
      * Muon (lr=0.005) for hidden weight matrices in ResidualBlocks
      * AdamW (lr=5e-4) for head projections, embeddings, norms
      * AdamW (lr=1e-5, wd=3e-4) for shallow backbone (mps.6, mps.7, post_mp)
      * AdamW (lr=2e-6, wd=3e-4) for deep backbone (mps.4, mps.5)
  - Class-weighted focal loss: gamma=2.0, weights=[2.0, 0.5, 4.0] for (down, neutral, up)
  - Label smoothing epsilon=0.10
  - SGDR (T_0=20, T_mult=2) — shorter cycles for richer checkpoint diversity
  - Quality-filtered top-8 SWA: only include checkpoints with val_f1 >= 0.498
  - Patience=150, max_epochs=400

Key differences from parent (node2-1-1-1-2-1-1-1, F1=0.5118):
  1. Deeper backbone: unfreeze mps.4, mps.5 at very low LR=2e-6
     (Sibling feedback: architecture saturation at head level; unlock deeper GNN layers)
  2. Quality-filtered top-8 SWA replacing 2-model post-hoc SWA
     (Sibling feedback Priority 1: uniform 23-ckpt SWA diluted by early poor checkpoints)
  3. SGDR T_0=20, T_mult=2 replacing single cosine (more warm restarts than sibling's T_0=40)
     (Sibling feedback Priority 2: shorter cycles generate more diverse quality checkpoints)
  4. Weight decay 4e-3 → 3e-3 (same relaxation as sibling; recovers Phase 2 exploration)
  5. max_epochs 300 → 400, patience 100 → 150 (allow full SGDR exploration)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import json
import argparse
import re
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
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# ─── Constants ────────────────────────────────────────────────────────────────

N_GENES_OUT = 6640
N_CLASSES = 3
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
LABEL_GENES_FILE = Path("data/label_genes.txt")

# Tree-validated class weights
CLASS_WEIGHTS_LIST = [2.0, 0.5, 4.0]  # [down, neutral, up]

# Deeper backbone: only freeze mps.0-3; train mps.4-7 + post_mp
N_FROZEN_LAYERS = 4  # was 6 in parent; unfreeze mps.4 and mps.5


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_logits_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Exact per-gene macro F1 matching calc_metric.py logic."""
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

    Stores precomputed frozen embeddings (output of mps.0-3) for fast batch
    retrieval. Trainable layers mps.4-7, post_mp run at forward time in the
    LightningModule using the full graph.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        frozen_embeddings: np.ndarray,
        node_name_to_idx: Dict[str, int],
        embed_dim: int = 256,
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.has_labels = has_labels

        n_samples = len(df)
        node_indices = np.full(n_samples, -1, dtype=np.int64)
        for i, pert_id in enumerate(self.pert_ids):
            if pert_id in node_name_to_idx:
                node_indices[i] = node_name_to_idx[pert_id]

        self.node_indices = torch.from_numpy(node_indices)

        embeddings = np.zeros((n_samples, embed_dim), dtype=np.float32)
        for i, idx in enumerate(node_indices):
            if idx >= 0:
                embeddings[i] = frozen_embeddings[idx]
        self.embeddings = torch.from_numpy(embeddings)

        in_vocab = [node_name_to_idx.get(p, -1) >= 0 for p in self.pert_ids]
        self.in_vocab = torch.tensor(in_vocab, dtype=torch.bool)

        if has_labels and "label" in df.columns:
            rows = []
            for lbl_str in df["label"]:
                rows.append([x + 1 for x in json.loads(lbl_str)])
            self.labels = torch.tensor(rows, dtype=torch.long)
        else:
            self.has_labels = False

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int):
        item = {
            "pert_id":   self.pert_ids[idx],
            "symbol":    self.symbols[idx],
            "embedding": self.embeddings[idx],
            "node_idx":  self.node_indices[idx],
            "in_vocab":  self.in_vocab[idx],
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule.

    Precomputes frozen mps.0-3 embeddings for all STRING nodes.
    Trainable layers mps.4-7+post_mp run online in LightningModule.
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
        if hasattr(self, "train_ds"):
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[DataModule] Computing intermediate embeddings through mps.{N_FROZEN_LAYERS-1} (frozen)...")
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", weights_only=False)

        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        backbone = backbone.to(device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        edge_index = graph["edge_index"].to(device)
        edge_weight = graph["edge_weight"].to(device) if graph.get("edge_weight") is not None else None

        with torch.no_grad():
            x = backbone.emb.weight  # [N_nodes, 256]
            for i in range(N_FROZEN_LAYERS):  # mps.0 through mps.3
                layer = backbone.mps[i]
                x_conv = layer.conv(x, edge_index, edge_weight)
                x_norm = layer.norm(x_conv)
                x_act = layer.act(x_norm)
                x = x + layer.dropout(x_act)

        frozen_embeddings = x.float().cpu().numpy()  # [N_nodes, 256]

        node_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(node_names)}

        self.frozen_embeddings = frozen_embeddings
        self.frozen_embeddings_tensor = torch.from_numpy(frozen_embeddings)
        self.node_name_to_idx = node_name_to_idx
        self.n_gnn_nodes = len(node_names)
        self.edge_index = graph["edge_index"]
        self.edge_weight = graph["edge_weight"] if graph.get("edge_weight") is not None else None

        del backbone
        torch.cuda.empty_cache()

        print(f"[DataModule] Frozen embeddings shape (mps.0-{N_FROZEN_LAYERS-1}): {frozen_embeddings.shape}")
        print(f"[DataModule] Trainable: mps.{N_FROZEN_LAYERS}-7, post_mp (online GNN forward)")

        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        n_train_cov = sum(p in node_name_to_idx for p in dfs["train"]["pert_id"])
        print(f"[DataModule] Coverage: {n_train_cov}/{len(dfs['train'])} train genes in STRING_GNN")

        embed_dim = frozen_embeddings.shape[1]
        self.train_ds = PerturbationDataset(dfs["train"], frozen_embeddings, node_name_to_idx, embed_dim, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   frozen_embeddings, node_name_to_idx, embed_dim, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  frozen_embeddings, node_name_to_idx, embed_dim, True)

        label_genes_path = self.data_dir / "label_genes.txt"
        self.label_gene_ids: List[str] = []
        if label_genes_path.exists():
            with open(label_genes_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        self.label_gene_ids.append(parts[0])
        print(f"[DataModule] Loaded {len(self.label_gene_ids)} label gene IDs")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.micro_batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=True, drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.micro_batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True, drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True, drop_last=False,
            persistent_workers=self.num_workers > 0,
        )


# ─── Model Components ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual MLP block with dropout=0.45."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.45):
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
    """Prediction head with bilinear interaction and pre-initialized output gene embeddings."""

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        dropout: float = 0.45,
        n_residual_layers: int = 6,
        out_gene_emb_init: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_genes_out = n_genes_out
        self.rank = rank

        self.input_norm = nn.LayerNorm(gnn_dim)
        self.oov_embedding = nn.Parameter(torch.zeros(gnn_dim))
        nn.init.normal_(self.oov_embedding, std=0.02)
        self.proj_in = nn.Linear(gnn_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, expand=4, dropout=dropout)
             for _ in range(n_residual_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * rank)
        self.out_gene_emb = nn.Parameter(torch.empty(n_genes_out, rank))
        self.head_dropout = nn.Dropout(dropout)
        self._init_weights(out_gene_emb_init)

    def _init_weights(self, out_gene_emb_init: Optional[np.ndarray]):
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)

        if out_gene_emb_init is not None and out_gene_emb_init.shape[0] == self.n_genes_out:
            gnn_dim = out_gene_emb_init.shape[1]
            if gnn_dim != self.rank:
                proj_matrix = np.random.randn(gnn_dim, self.rank).astype(np.float32)
                proj_matrix /= np.sqrt(gnn_dim)
                projected = out_gene_emb_init @ proj_matrix
            else:
                projected = out_gene_emb_init.copy()
            projected = projected / (projected.std() + 1e-8) * 0.02
            self.out_gene_emb.data.copy_(torch.from_numpy(projected))
        else:
            nn.init.normal_(self.out_gene_emb, std=0.02)

    def forward(self, gnn_emb: torch.Tensor, in_vocab: torch.Tensor) -> torch.Tensor:
        oov_mask = ~in_vocab
        if oov_mask.any():
            B = gnn_emb.shape[0]
            oov_emb = self.oov_embedding.to(dtype=gnn_emb.dtype).unsqueeze(0).expand(B, -1)
            gnn_emb = gnn_emb.clone()
            gnn_emb[oov_mask] = oov_emb[oov_mask]

        x = self.input_norm(gnn_emb)
        x = self.proj_in(x)
        for blk in self.res_blocks:
            x = blk(x)
        x = self.norm_out(x)
        x = self.head_dropout(x)
        x = self.proj_bilinear(x)  # [B, 3*512]
        pert_proj = x.view(-1, self.n_classes, self.rank)  # [B, 3, 512]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)
        return logits


class PartialBackboneAdapter(nn.Module):
    """Wraps trainable STRING_GNN layers mps.4-7 + post_mp.

    Two-tier LR in the optimizer:
      - layer4, layer5: deep backbone at backbone_lr2=2e-6
      - layer6, layer7, post_mp: shallow backbone at backbone_lr=1e-5
    """

    def __init__(self, layer4, layer5, layer6, layer7, post_mp):
        super().__init__()
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
        self.layer7 = layer7
        self.post_mp = post_mp

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for layer in (self.layer4, self.layer5, self.layer6, self.layer7):
            x = x + layer.dropout(layer.act(
                layer.norm(layer.conv(x, edge_index, edge_weight))
            ))
        x = self.post_mp(x)
        return x


# ─── Focal Loss with Label Smoothing ──────────────────────────────────────────

def focal_loss_with_label_smoothing(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.10,
) -> torch.Tensor:
    B, C, G = logits.shape
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)
    labels_flat = labels.reshape(-1)

    if label_smoothing > 0.0:
        one_hot = F.one_hot(labels_flat, num_classes=C).float()
        smooth_targets = (1.0 - label_smoothing) * one_hot + label_smoothing / C
        log_probs = F.log_softmax(logits_flat, dim=1)
        if class_weights is not None:
            weighted = smooth_targets * class_weights.unsqueeze(0)
            ce_loss = -(weighted * log_probs).sum(dim=1)
        else:
            ce_loss = -(smooth_targets * log_probs).sum(dim=1)
        with torch.no_grad():
            probs = torch.softmax(logits_flat, dim=1)
            pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
    else:
        ce_loss = F.cross_entropy(logits_flat, labels_flat, weight=class_weights, reduction="none")
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
    dist.all_gather(g_preds, p)
    dist.all_gather(g_labels, l)

    real_preds  = torch.cat([g_preds[i][:all_sizes[i].item()].cpu()  for i in range(world_size)], 0)
    real_labels = torch.cat([g_labels[i][:all_sizes[i].item()].cpu() for i in range(world_size)], 0)
    return real_preds, real_labels


# ─── Quality-Filtered SWA Inference ───────────────────────────────────────────

def _parse_val_f1_from_filename(ckpt_path: Path) -> Optional[float]:
    """Extract val_f1 from checkpoint filename like 'periodic-0050-val_f1=0.5080.ckpt'."""
    m = re.search(r"val_f1=([0-9.]+)", ckpt_path.name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _parse_epoch_from_filename(ckpt_path: Path) -> Optional[int]:
    """Extract epoch from checkpoint filename."""
    m = re.search(r"(?:periodic|best|epoch=?)[=-](\d+)", ckpt_path.name)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def build_quality_filtered_swa(
    lit: "PerturbationLitModule",
    periodic_ckpt_dir: Path,
    best_ckpt_path: Optional[str],
    last_ckpt_path: Optional[str],
    swa_start_epoch: int,
    swa_top_k: int,
    swa_min_val_f1: float,
    device: torch.device,
) -> Optional[AveragedModel]:
    """Build a quality-filtered SWA model from periodic checkpoints.

    Selects top-K checkpoints by val_f1 where val_f1 >= swa_min_val_f1
    and epoch >= swa_start_epoch.
    """
    candidates: List[Tuple[float, Path]] = []  # (val_f1, path)

    # Collect periodic checkpoints
    if periodic_ckpt_dir.exists():
        for ckpt_path in sorted(periodic_ckpt_dir.glob("*.ckpt")):
            val_f1 = _parse_val_f1_from_filename(ckpt_path)
            epoch = _parse_epoch_from_filename(ckpt_path)
            if val_f1 is not None and epoch is not None and epoch >= swa_start_epoch:
                if val_f1 >= swa_min_val_f1:
                    candidates.append((val_f1, ckpt_path))

    # Add best checkpoint if not already included
    if best_ckpt_path and Path(best_ckpt_path).exists():
        best_val_f1 = _parse_val_f1_from_filename(Path(best_ckpt_path))
        if best_val_f1 is not None and best_val_f1 >= swa_min_val_f1:
            candidates.append((best_val_f1, Path(best_ckpt_path)))

    # Deduplicate by path
    seen_paths: set = set()
    unique_candidates: List[Tuple[float, Path]] = []
    for val_f1, path in candidates:
        rp = str(path.resolve())
        if rp not in seen_paths:
            seen_paths.add(rp)
            unique_candidates.append((val_f1, path))

    # Sort by val_f1 descending, keep top-K
    unique_candidates.sort(key=lambda x: x[0], reverse=True)
    selected = unique_candidates[:swa_top_k]

    if len(selected) < 1:
        print(f"[SWA] No qualifying checkpoints found (min_val_f1={swa_min_val_f1}). "
              f"Found {len(unique_candidates)} total periodic checkpoints.")
        # Fall back to all candidates regardless of threshold
        all_cands: List[Tuple[float, Path]] = []
        if periodic_ckpt_dir.exists():
            for ckpt_path in sorted(periodic_ckpt_dir.glob("*.ckpt")):
                val_f1 = _parse_val_f1_from_filename(ckpt_path)
                epoch = _parse_epoch_from_filename(ckpt_path)
                if val_f1 is not None and epoch is not None and epoch >= swa_start_epoch:
                    all_cands.append((val_f1, ckpt_path))
        all_cands.sort(key=lambda x: x[0], reverse=True)
        selected = all_cands[:swa_top_k]
        if not selected:
            return None

    print(f"[SWA] Quality-filtered SWA: {len(selected)}/{len(unique_candidates)} qualifying checkpoints")
    for rank_i, (vf1, p) in enumerate(selected):
        print(f"[SWA]   [{rank_i+1}] val_f1={vf1:.4f} — {p.name}")

    # Average with AveragedModel
    lit = lit.to(device)
    if lit.frozen_node_embs is not None:
        lit.frozen_node_embs = lit.frozen_node_embs.to(device)
    lit.edge_index_buf = lit.edge_index_buf.to(device)
    if lit.edge_weight_buf is not None:
        lit.edge_weight_buf = lit.edge_weight_buf.to(device)

    swa_model: Optional[AveragedModel] = None
    for _, ckpt_path in selected:
        ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = ckpt_data.get("state_dict", ckpt_data)
        lit.load_state_dict(state)
        lit.eval()
        if swa_model is None:
            swa_model = AveragedModel(lit)
        swa_model.update_parameters(lit)

    return swa_model


def run_swa_inference(
    swa_model: AveragedModel,
    test_ds: "PerturbationDataset",
    lit: "PerturbationLitModule",
    micro_batch_size: int,
    num_workers: int,
    out_dir: Path,
) -> None:
    """Run inference using quality-filtered SWA model. Overwrites test_predictions.tsv."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_loader = DataLoader(
        test_ds, batch_size=micro_batch_size * 2,
        shuffle=False, num_workers=min(num_workers, 4),
        pin_memory=True, drop_last=False, persistent_workers=False,
    )

    swa_model.eval()
    swa_model = swa_model.to(device)
    module = swa_model.module
    if hasattr(module, "frozen_node_embs") and module.frozen_node_embs is not None:
        module.frozen_node_embs = module.frozen_node_embs.to(device)
    if hasattr(module, "edge_index_buf"):
        module.edge_index_buf = module.edge_index_buf.to(device)
    if hasattr(module, "edge_weight_buf") and module.edge_weight_buf is not None:
        module.edge_weight_buf = module.edge_weight_buf.to(device)

    print("[SWA] Running inference with quality-filtered SWA model...")

    probs_list: List[torch.Tensor] = []
    pert_ids_list: List[str] = []
    syms_list: List[str] = []

    with torch.no_grad():
        for batch in test_loader:
            emb = batch["embedding"].float().to(device)
            node_idx = batch["node_idx"].to(device)
            in_vocab = batch["in_vocab"].to(device)
            final_emb = module._get_final_embeddings(emb, node_idx, in_vocab)
            logits = module.head(final_emb, in_vocab)
            probs = torch.softmax(logits, dim=1).float().cpu()
            probs_list.append(probs)
            pert_ids_list.extend(batch["pert_id"])
            syms_list.extend(batch["symbol"])

    all_probs = torch.cat(probs_list, dim=0).numpy()
    print(f"[SWA] SWA inference complete: {all_probs.shape[0]} samples")

    pred_path = out_dir / "test_predictions.tsv"
    seen_ids: set = set()
    with open(pred_path, "w") as fh:
        fh.write("idx\tinput\tprediction\n")
        for pert_id, symbol, probs in zip(pert_ids_list, syms_list, all_probs):
            if pert_id not in seen_ids:
                seen_ids.add(pert_id)
                fh.write(f"{pert_id}\t{symbol}\t{json.dumps(probs.tolist())}\n")

    print(f"[SWA] Saved {len(seen_ids)} quality-filtered SWA predictions → {pred_path}")


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbationLitModule(pl.LightningModule):
    """Gene-perturbation DEG prediction with deeper backbone fine-tuning.

    Architecture changes from parent:
    - N_FROZEN_LAYERS=4: mps.0-3 frozen, mps.4-7+post_mp trainable
    - Two-tier backbone LR: mps.4,5 at backbone_lr2=2e-6; mps.6,7,post_mp at backbone_lr=1e-5
    - SGDR T_0=20, T_mult=2 for LR scheduling
    - Quality-filtered top-K SWA for test-time inference
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_residual_layers: int = 6,
        dropout: float = 0.45,
        lr_muon: float = 0.005,
        lr_adamw: float = 5e-4,
        backbone_lr: float = 1e-5,
        backbone_lr2: float = 2e-6,
        weight_decay: float = 3e-3,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.10,
        use_class_weights: bool = True,
        warmup_steps: int = 100,
        sgdr_t0_steps: int = 440,       # T_0=20 epochs × 22 eff_steps_per_epoch
        sgdr_t_mult: float = 2.0,
        sgdr_eta_min: float = 1e-6,
        swa_start_epoch: int = 30,
        swa_top_k: int = 8,
        swa_min_val_f1: float = 0.498,
        out_gene_emb_init: Optional[np.ndarray] = None,
        frozen_embeddings_np: Optional[np.ndarray] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["out_gene_emb_init", "frozen_embeddings_np",
                                          "edge_index", "edge_weight"])
        self._out_gene_emb_init = out_gene_emb_init
        self._frozen_embeddings_np = frozen_embeddings_np
        self._edge_index = edge_index
        self._edge_weight = edge_weight

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams

        self.head = GNNBilinearHead(
            gnn_dim=hp.gnn_dim,
            hidden_dim=hp.hidden_dim,
            rank=hp.rank,
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            dropout=hp.dropout,
            n_residual_layers=hp.n_residual_layers,
            out_gene_emb_init=self._out_gene_emb_init,
        )

        # Build deeper backbone adapter (mps.4-7 + post_mp)
        full_backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        self.backbone_adapter = PartialBackboneAdapter(
            layer4=full_backbone.mps[4],
            layer5=full_backbone.mps[5],
            layer6=full_backbone.mps[6],
            layer7=full_backbone.mps[7],
            post_mp=full_backbone.post_mp,
        )
        del full_backbone
        deep_params = sum(p.numel() for n, p in self.backbone_adapter.named_parameters()
                          if "layer4" in n or "layer5" in n)
        shallow_params = sum(p.numel() for n, p in self.backbone_adapter.named_parameters()
                             if "layer6" in n or "layer7" in n or "post_mp" in n)
        print(f"[Setup] Deep backbone (mps.4,5): {deep_params:,} params @ lr={hp.backbone_lr2}")
        print(f"[Setup] Shallow backbone (mps.6,7,post_mp): {shallow_params:,} params @ lr={hp.backbone_lr}")

        if self._frozen_embeddings_np is not None:
            frozen_emb_tensor = torch.from_numpy(self._frozen_embeddings_np.astype(np.float32))
            self.register_buffer("frozen_node_embs", frozen_emb_tensor)
        else:
            print("[Setup] WARNING: No frozen embeddings provided")
            self.frozen_node_embs = None

        if self._edge_index is not None:
            self.register_buffer("edge_index_buf", self._edge_index)
        else:
            self.register_buffer("edge_index_buf", torch.zeros(2, 1, dtype=torch.long))
        if self._edge_weight is not None:
            self.register_buffer("edge_weight_buf", self._edge_weight)
        else:
            self.register_buffer("edge_weight_buf", None)

        if hp.use_class_weights:
            cw = torch.tensor(CLASS_WEIGHTS_LIST, dtype=torch.float32)
            self.register_buffer("class_weights", cw)
            print(f"[Setup] Using class weights [2.0, 0.5, 4.0]")
        else:
            self.class_weights = None

        for p in self.head.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        for p in self.backbone_adapter.parameters():
            p.data = p.data.float()

        head_trainable = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        backbone_trainable = sum(p.numel() for p in self.backbone_adapter.parameters())
        print(f"[Setup] Head trainable: {head_trainable:,} | Backbone trainable: {backbone_trainable:,}")

    def _get_final_embeddings(
        self,
        batch_embeddings: torch.Tensor,
        node_indices: torch.Tensor,
        in_vocab: torch.Tensor,
    ) -> torch.Tensor:
        if self.frozen_node_embs is None:
            return batch_embeddings

        full_final_embs = self.backbone_adapter(
            self.frozen_node_embs.float(),
            self.edge_index_buf,
            self.edge_weight_buf,
        )

        B = batch_embeddings.shape[0]
        device = batch_embeddings.device
        result = torch.zeros(B, batch_embeddings.shape[1], device=device, dtype=torch.float32)

        in_vocab_mask = in_vocab & (node_indices >= 0)
        if in_vocab_mask.any():
            valid_indices = node_indices[in_vocab_mask]
            result[in_vocab_mask] = full_final_embs[valid_indices].float()

        return result

    def forward(self, embedding, node_indices, in_vocab):
        final_emb = self._get_final_embeddings(embedding, node_indices, in_vocab)
        return self.head(final_emb, in_vocab)

    def _compute_loss(self, logits, labels):
        cw = None
        if self.hparams.use_class_weights and hasattr(self, "class_weights") and self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
        return focal_loss_with_label_smoothing(
            logits, labels,
            gamma=self.hparams.focal_gamma,
            class_weights=cw,
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["embedding"].float(), batch["node_idx"], batch["in_vocab"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["embedding"].float(), batch["node_idx"], batch["in_vocab"])
        if "label" in batch:
            loss = self._compute_loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())
        return logits

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        local_p = torch.cat(self._val_preds, dim=0)
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
        logits = self(batch["embedding"].float(), batch["node_idx"], batch["in_vocab"])
        probs = torch.softmax(logits, dim=1)
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
            all_probs, all_labels = _gather_tensors(local_probs, dummy_labels, self.device, self.trainer.world_size)
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

        if self.trainer.is_global_zero:
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"
            seen_ids: set = set()
            dedup_probs: list = []
            dedup_labels: list = []
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for i, (pert_id, symbol, probs) in enumerate(
                    zip(all_pert, all_syms, all_probs.numpy())
                ):
                    if pert_id not in seen_ids:
                        seen_ids.add(pert_id)
                        fh.write(f"{pert_id}\t{symbol}\t{json.dumps(probs.tolist())}\n")
                        dedup_probs.append(probs)
                        dedup_labels.append(all_labels[i].numpy())

            self.print(f"[Node2-1-1-1-2-1-1-1-2] Saved test predictions → {pred_path} ({len(seen_ids)} samples)")

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node2-1-1-1-2-1-1-1-2] Self-computed test F1 (standard ckpt) = {f1:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        muon_params = []
        adamw_head_params = []
        adamw_shallow_backbone_params = []
        adamw_deep_backbone_params = []

        for name, param in self.head.named_parameters():
            if not param.requires_grad:
                continue
            is_hidden_matrix = (
                param.ndim >= 2
                and "res_blocks" in name
                and "weight" in name
                and "norm" not in name
            )
            if is_hidden_matrix:
                muon_params.append(param)
            else:
                adamw_head_params.append(param)

        for name, param in self.backbone_adapter.named_parameters():
            if not param.requires_grad:
                continue
            if "layer4" in name or "layer5" in name:
                adamw_deep_backbone_params.append(param)
            else:
                adamw_shallow_backbone_params.append(param)

        n_muon    = sum(p.numel() for p in muon_params)
        n_head    = sum(p.numel() for p in adamw_head_params)
        n_shallow = sum(p.numel() for p in adamw_shallow_backbone_params)
        n_deep    = sum(p.numel() for p in adamw_deep_backbone_params)
        print(f"[Optimizer] Muon={n_muon:,} | AdamW head={n_head:,} | "
              f"shallow_backbone={n_shallow:,} | deep_backbone={n_deep:,}")

        try:
            from muon import MuonWithAuxAdam

            param_groups = [
                dict(params=muon_params,
                     use_muon=True, lr=hp.lr_muon,
                     weight_decay=hp.weight_decay, momentum=0.95),
                dict(params=adamw_head_params,
                     use_muon=False, lr=hp.lr_adamw,
                     betas=(0.9, 0.95), eps=1e-10,
                     weight_decay=hp.weight_decay),
                dict(params=adamw_shallow_backbone_params,
                     use_muon=False, lr=hp.backbone_lr,
                     betas=(0.9, 0.95), eps=1e-10,
                     weight_decay=hp.weight_decay * 0.1),
                dict(params=adamw_deep_backbone_params,
                     use_muon=False, lr=hp.backbone_lr2,
                     betas=(0.9, 0.95), eps=1e-10,
                     weight_decay=hp.weight_decay * 0.1),
            ]
            optimizer = MuonWithAuxAdam(param_groups)
            print(f"[Optimizer] MuonWithAuxAdam: lr_muon={hp.lr_muon}, "
                  f"lr_head={hp.lr_adamw}, lr_shallow={hp.backbone_lr}, "
                  f"lr_deep={hp.backbone_lr2}, wd={hp.weight_decay}")

        except ImportError:
            print("[Optimizer] WARNING: MuonWithAuxAdam not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(
                [
                    {"params": muon_params,                   "lr": hp.lr_adamw},
                    {"params": adamw_head_params,             "lr": hp.lr_adamw},
                    {"params": adamw_shallow_backbone_params, "lr": hp.backbone_lr},
                    {"params": adamw_deep_backbone_params,    "lr": hp.backbone_lr2},
                ],
                weight_decay=hp.weight_decay,
            )

        # SGDR step-level lambda with linear warmup
        sgdr_t0   = hp.sgdr_t0_steps
        sgdr_mult = hp.sgdr_t_mult
        eta_min   = hp.sgdr_eta_min

        def lr_lambda(step: int) -> float:
            if step < hp.warmup_steps:
                return float(step) / max(1, hp.warmup_steps)
            t = step - hp.warmup_steps
            t_i = sgdr_t0
            while t >= t_i:
                t -= t_i
                t_i = max(1, int(round(t_i * sgdr_mult)))
            progress = t / max(1, t_i)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return eta_min + (1.0 - eta_min) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        trainable_sd = {k: v for k, v in full_sd.items() if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving ckpt: {trained}/{total} trainable params ({100*trained/total:.1f}%)")
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 2-1-1-1-2-1-1-1-2 – Deeper Backbone (mps.4-7) + "
                    "Quality-Filtered Top-8 SWA + SGDR T_0=20 + WD=3e-3"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--gnn-dim",            type=int,   default=256)
    p.add_argument("--hidden-dim",         type=int,   default=512)
    p.add_argument("--rank",               type=int,   default=512)
    p.add_argument("--n-residual-layers",  type=int,   default=6)
    p.add_argument("--dropout",            type=float, default=0.45)
    p.add_argument("--lr-muon",            type=float, default=0.005)
    p.add_argument("--lr-adamw",           type=float, default=5e-4)
    p.add_argument("--backbone-lr",        type=float, default=1e-5,
                   help="LR for shallow backbone layers (mps.6, mps.7, post_mp)")
    p.add_argument("--backbone-lr2",       type=float, default=2e-6,
                   help="LR for deep backbone layers (mps.4, mps.5) — very conservative")
    p.add_argument("--weight-decay",       type=float, default=3e-3,
                   help="Weight decay (reduced from parent 4e-3; recovers Phase 2 exploration)")
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--label-smoothing",    type=float, default=0.10)
    p.add_argument("--use-class-weights",  action="store_true", default=True)
    p.add_argument("--no-class-weights",   dest="use_class_weights", action="store_false")
    p.add_argument("--warmup-steps",       type=int,   default=100)
    p.add_argument("--sgdr-t0",            type=int,   default=20,
                   help="SGDR first cycle length in epochs")
    p.add_argument("--sgdr-t-mult",        type=float, default=2.0)
    p.add_argument("--sgdr-eta-min",       type=float, default=1e-6)
    p.add_argument("--swa-start-epoch",    type=int,   default=30,
                   help="Epoch from which periodic checkpoints qualify for SWA pool")
    p.add_argument("--swa-top-k",          type=int,   default=8,
                   help="Max number of checkpoints to average (quality-filtered top-K)")
    p.add_argument("--swa-min-val-f1",     type=float, default=0.498,
                   help="Minimum val_f1 for a checkpoint to qualify for SWA pool")
    p.add_argument("--swa-every-n-epochs", type=int,   default=10,
                   help="Periodic checkpoint frequency in epochs")
    p.add_argument("--use-swa",            action="store_true", default=True)
    p.add_argument("--no-swa",             dest="use_swa", action="store_false")
    p.add_argument("--micro-batch-size",   type=int,   default=16)
    p.add_argument("--global-batch-size",  type=int,   default=64)
    p.add_argument("--max-epochs",         type=int,   default=400)
    p.add_argument("--patience",           type=int,   default=150)
    p.add_argument("--num-workers",        type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",     type=int,   default=None)
    p.add_argument("--fast-dev-run",       action="store_true", default=False)
    return p.parse_args()


def build_out_gene_emb_init(
    frozen_embeddings: np.ndarray,
    node_name_to_idx: Dict[str, int],
    label_gene_ids: List[str],
    rank: int,
) -> Optional[np.ndarray]:
    gnn_dim = frozen_embeddings.shape[1]
    out_init = np.random.randn(len(label_gene_ids), gnn_dim).astype(np.float32) * 0.02

    n_found = 0
    for i, gene_id in enumerate(label_gene_ids):
        if gene_id in node_name_to_idx:
            out_init[i] = frozen_embeddings[node_name_to_idx[gene_id]]
            n_found += 1

    print(f"[Init] Output gene embeddings: {n_found}/{len(label_gene_ids)} "
          f"({100*n_found/len(label_gene_ids):.1f}%) initialized from STRING_GNN mps.0-{N_FROZEN_LAYERS-1}")
    return out_init


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    dm = PerturbationDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    out_gene_emb_init = None
    if hasattr(dm, "label_gene_ids") and len(dm.label_gene_ids) == N_GENES_OUT:
        out_gene_emb_init = build_out_gene_emb_init(
            dm.frozen_embeddings,
            dm.node_name_to_idx,
            dm.label_gene_ids,
            rank=args.rank,
        )
    else:
        print("[Main] WARNING: label_gene_ids not loaded or wrong length")

    # Calibrate SGDR T_0 in steps
    steps_per_epoch = max(1, len(dm.train_ds) // (args.micro_batch_size * n_gpus))
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)
    sgdr_t0_steps = effective_steps_per_epoch * args.sgdr_t0

    print(f"[Main] effective_steps_per_epoch={effective_steps_per_epoch}")
    print(f"[Main] SGDR T_0={args.sgdr_t0} epochs = {sgdr_t0_steps} steps, T_mult={args.sgdr_t_mult}")
    print(f"[Main] weight_decay={args.weight_decay} (reduced from parent 4e-3)")
    print(f"[Main] dropout={args.dropout}, backbone_lr={args.backbone_lr}, backbone_lr2={args.backbone_lr2}")
    print(f"[Main] N_FROZEN_LAYERS={N_FROZEN_LAYERS} (mps.4,5 now trainable)")

    lit = PerturbationLitModule(
        gnn_dim=args.gnn_dim,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
        n_residual_layers=args.n_residual_layers,
        dropout=args.dropout,
        lr_muon=args.lr_muon,
        lr_adamw=args.lr_adamw,
        backbone_lr=args.backbone_lr,
        backbone_lr2=args.backbone_lr2,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        use_class_weights=args.use_class_weights,
        warmup_steps=args.warmup_steps,
        sgdr_t0_steps=sgdr_t0_steps,
        sgdr_t_mult=args.sgdr_t_mult,
        sgdr_eta_min=args.sgdr_eta_min,
        swa_start_epoch=args.swa_start_epoch,
        swa_top_k=args.swa_top_k,
        swa_min_val_f1=args.swa_min_val_f1,
        out_gene_emb_init=out_gene_emb_init,
        frozen_embeddings_np=dm.frozen_embeddings,
        edge_index=dm.edge_index,
        edge_weight=dm.edge_weight,
    )

    # Primary: best val_f1 checkpoint
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-val_f1={val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=1,
        save_last=True,
    )

    # Periodic: every N epochs for quality-filtered SWA pool
    periodic_ckpt_dir = out_dir / "checkpoints" / "periodic"
    periodic_ckpt_dir.mkdir(parents=True, exist_ok=True)
    periodic_ckpt_cb = ModelCheckpoint(
        dirpath=str(periodic_ckpt_dir),
        filename="periodic-{epoch:04d}-val_f1={val_f1:.4f}",
        monitor="val_f1",
        save_top_k=-1,           # save all
        every_n_epochs=args.swa_every_n_epochs,
        save_last=False,
    )

    es_cb  = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-5)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

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

    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

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
        callbacks=[ckpt_cb, periodic_ckpt_cb, es_cb, lr_cb, pb_cb],
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

    # Quality-filtered top-K SWA (rank 0 only)
    if (trainer.is_global_zero
            and args.use_swa
            and not args.fast_dev_run
            and args.debug_max_step is None):
        try:
            print(f"[SWA] Building quality-filtered top-{args.swa_top_k} SWA model "
                  f"(min_val_f1={args.swa_min_val_f1})...")

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load best checkpoint as starting state for lit module
            best_ckpt_path = ckpt_cb.best_model_path
            if best_ckpt_path and Path(best_ckpt_path).exists():
                ckpt_data = torch.load(str(best_ckpt_path), map_location="cpu", weights_only=False)
                lit.load_state_dict(ckpt_data.get("state_dict", ckpt_data))
                lit.eval()

            swa_model = build_quality_filtered_swa(
                lit=lit,
                periodic_ckpt_dir=periodic_ckpt_dir,
                best_ckpt_path=ckpt_cb.best_model_path,
                last_ckpt_path=ckpt_cb.last_model_path,
                swa_start_epoch=args.swa_start_epoch,
                swa_top_k=args.swa_top_k,
                swa_min_val_f1=args.swa_min_val_f1,
                device=device,
            )

            if swa_model is not None:
                run_swa_inference(
                    swa_model=swa_model,
                    test_ds=dm.test_ds,
                    lit=lit,
                    micro_batch_size=args.micro_batch_size,
                    num_workers=args.num_workers,
                    out_dir=out_dir,
                )
            else:
                print("[SWA] No qualified checkpoints, keeping single-checkpoint predictions")

        except Exception as e:
            print(f"[SWA] WARNING: quality-filtered SWA failed: {e}")
            import traceback
            traceback.print_exc()
            print("[SWA] Falling back to single-checkpoint predictions (already saved)")

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"Node 2-1-1-1-2-1-1-1-2 – Deeper Backbone (mps.4-7) + Quality-Filtered Top-8 SWA "
            f"+ SGDR T_0=20 + WD=3e-3 + Dropout=0.45\n"
            f"Test results from trainer: {test_results}\n"
            f"(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
