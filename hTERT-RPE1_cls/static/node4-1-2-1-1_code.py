"""
Node 4-1-2-1-1 – STRING_GNN Partial Fine-Tuning (mps.6+7+post_mp) +
              Rank-512 Deep Residual Bilinear Head (6 ResBlocks) +
              MuonWithAuxAdam + Cosine Warm Restarts (SGDR) +
              Quality-Filtered Weighted SWA + Corrected Schedule

This node directly addresses the over-regularization regression in node4-1-2-1
(F1=0.4972) by:

Key changes vs node4-1-2-1 (parent):
  1. N_RESBLOCKS restored 4 → 6 (recover head capacity; 6-ResBlock head reliably reaches ~0.506)
  2. Unfreezes mps.6 in addition to mps.7+post_mp (~198K backbone params vs ~67K)
  3. Dropout adjusted 0.45 → 0.40 (intermediate; less aggressive than parent)
  4. label_smoothing reduced 0.05 → 0.02 (very light smoothing only)
  5. Patience increased 30 → 50 (allow deeper staircase cycles to develop)
  6. max_epochs increased 150 → 200 (matched to patience)
  7. Added quality-filtered exponentially-weighted SWA post-hoc averaging
     (top_k=15, temperature=3.0, val_f1_threshold=0.490)
  8. Periodic checkpoints every 5 epochs for SWA pool collection
  9. Precomputed cache now runs through mps.5 (layers 0-5) instead of mps.6

Architecture:
  - STRING_GNN loaded from /home/Models/STRING_GNN
  - Frozen: emb.weight + mps.0-5 (precomputed as fixed buffer after mps.5)
  - Trainable backbone: mps.6 + mps.7 + post_mp (~198K params at lr=1e-5)
  - 6-layer deep residual bilinear MLP head (restored from node4-1-2):
      InputProj: Linear(256 → 512) + GELU
      6x ResidualBlock: [LN → Linear(512→2048) → GELU → Dropout(0.40) → Linear(2048→512) + skip]
      BilinearProj: Linear(512 → 3×512) → reshape [B, 3, 512]
      out_gene_emb: Embedding(6640, 512) learnable
      logits: [B, 3, 512] @ [512, 6640] → [B, 3, 6640]
  - Optimizer: MuonWithAuxAdam with 3 parameter groups:
      Group 1 (Muon): ResBlock 2D weight matrices, lr=0.005, wd=1e-4
      Group 2 (AdamW): head scalars/biases/embeddings/LayerNorms, lr=5e-4, wd=1e-3
      Group 3 (AdamW): backbone mps.6+7+post_mp params, lr=1e-5, wd=0
  - LR Schedule: CosineAnnealingWarmRestarts (T_0=300 steps, T_mult=1, eta_min=1e-7)
  - Loss: Focal cross-entropy (gamma=2.0, label_smoothing=0.02, class_weights=[2.0, 0.5, 4.0])
  - Post-hoc SWA: Quality-filtered top-k weighted average at test time
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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# ─── Constants ────────────────────────────────────────────────────────────────

STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT    = 6640
N_CLASSES      = 3
GNN_DIM        = 256
HEAD_DIM       = 512
HEAD_EXPAND    = 4
N_RESBLOCKS    = 6   # Restored from 4 to recover full head capacity

# Number of frozen GNN layers to precompute (0-5 frozen, 6-7+post_mp trainable)
N_FROZEN_LAYERS = 6   # Layers 0 through 5 are frozen (was 7 in parent)


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Mirrors calc_metric.py: per-gene macro F1 averaged over all genes."""
    pred_cls = pred_np.argmax(axis=1)  # [B, G]
    f1_vals  = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]
        yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1   = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Focal CE with optional label smoothing.

    When label_smoothing > 0, distributes smoothing_mass / n_cls probability
    uniformly and (1 - smoothing_mass) probability on the hard target,
    then applies focal weighting based on the hard-target probability.
    """
    log_probs = F.log_softmax(logits, dim=-1)   # [B*G, C]
    probs     = log_probs.exp()

    if label_smoothing > 0.0:
        n_cls    = logits.shape[-1]
        smooth   = label_smoothing / n_cls
        # Smooth CE: uniform component + hard component
        ce_loss  = -(log_probs * smooth).sum(dim=-1)  # uniform
        ce_hard  = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        ce_loss  = ce_loss + (1.0 - label_smoothing) * ce_hard
        if class_weights is not None:
            ce_loss = ce_loss * class_weights[targets]
    else:
        ce_loss = F.nll_loss(log_probs, targets, weight=class_weights, reduction='none')

    p_t     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_w = (1.0 - p_t) ** gamma
    return (focal_w * ce_loss).mean()


# ─── Dataset ──────────────────────────────────────────────────────────────────

class StringGNNDataset(Dataset):
    """Stores pert_id → STRING_GNN node index mapping and labels."""

    def __init__(
        self,
        pert_ids:     List[str],
        symbols:      List[str],
        node_indices: torch.Tensor,              # [N] long, -1 for unknown
        labels:       Optional[torch.Tensor] = None,  # [N, 6640] long
    ):
        self.pert_ids     = pert_ids
        self.symbols      = symbols
        self.node_indices = node_indices
        self.labels       = labels

    def __len__(self): return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
            "node_index": self.node_indices[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    out = {
        "pert_id":    [b["pert_id"]    for b in batch],
        "symbol":     [b["symbol"]     for b in batch],
        "node_index": torch.stack([b["node_index"] for b in batch]),
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class StringGNNDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data", micro_batch_size=4, num_workers=4):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage=None):
        gnn_dir    = Path(STRING_GNN_DIR)
        node_names = json.loads((gnn_dir / "node_names.json").read_text())
        node_name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(node_names)}

        def load_split(fname: str, has_lbl: bool):
            df   = pd.read_csv(self.data_dir / fname, sep="\t")
            idxs = torch.tensor(
                [node_name_to_idx.get(pid, -1) for pid in df["pert_id"].tolist()],
                dtype=torch.long,
            )
            labels = None
            if has_lbl and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return StringGNNDataset(
                df["pert_id"].tolist(), df["symbol"].tolist(), idxs, labels
            )

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

        graph = torch.load(gnn_dir / "graph_data.pt", weights_only=False)
        self.edge_index  = graph["edge_index"]
        self.edge_weight = graph.get("edge_weight", None)
        self.n_nodes     = len(node_names)  # 18870

        # Coverage check
        n_unknown = sum(
            1 for ds in (self.train_ds, self.val_ds, self.test_ds)
            for ni in ds.node_indices.tolist() if ni == -1
        )
        total = len(self.train_ds) + len(self.val_ds) + len(self.test_ds)
        print(f"[Node4-1-2-1-1] {n_unknown}/{total} samples not found in STRING_GNN graph "
              f"→ will use learned fallback embedding.")

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds, batch_size=self.micro_batch_size, shuffle=shuffle,
            collate_fn=collate_fn, num_workers=self.num_workers,
            pin_memory=True, drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Model Components ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Standard residual block: LN → Linear(d→d*expand) → GELU → Dropout → Linear(d*expand→d) + skip."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.40):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.fc1   = nn.Linear(dim, dim * expand, bias=True)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.fc2   = nn.Linear(dim * expand, dim, bias=False)

    def forward(self, x):
        h = self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))
        return x + h


class PartialGNNWithHead(nn.Module):
    """
    STRING_GNN backbone where:
      - emb.weight + mps.0-5 are FROZEN (precomputed & cached)
      - mps.6 + mps.7 + post_mp are TRAINABLE at a low LR

    The frozen portion (layers 0-5) is precomputed once and stored as a buffer.
    Only the last two GNN layers (mps.6, mps.7) run per forward pass, then post_mp.
    This gives ~198K trainable backbone params vs ~67K in the parent.
    """

    def __init__(
        self,
        edge_index:   torch.Tensor,
        edge_weight:  Optional[torch.Tensor],
        n_nodes:      int,
        n_genes_out:  int   = N_GENES_OUT,
        n_classes:    int   = N_CLASSES,
        gnn_dim:      int   = GNN_DIM,
        head_dim:     int   = HEAD_DIM,
        head_expand:  int   = HEAD_EXPAND,
        n_resblocks:  int   = N_RESBLOCKS,
        dropout:      float = 0.40,
    ):
        super().__init__()
        self.n_nodes      = n_nodes
        self.gnn_dim      = gnn_dim
        self.head_dim     = head_dim
        self.n_frozen     = N_FROZEN_LAYERS  # 6 layers frozen (0-5)

        # ── Load full STRING_GNN ──────────────────────────────────────────
        self.gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)

        # Freeze everything first, then selectively unfreeze mps.6, mps.7 and post_mp
        for p in self.gnn.parameters():
            p.requires_grad_(False)
        for p in self.gnn.mps[6].parameters():
            p.requires_grad_(True)
        for p in self.gnn.mps[7].parameters():
            p.requires_grad_(True)
        for p in self.gnn.post_mp.parameters():
            p.requires_grad_(True)

        # Register graph tensors as buffers for auto device transfer
        self.register_buffer("edge_index", edge_index)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight = None

        # Buffer to cache frozen intermediate embeddings (filled during setup)
        # Shape: [n_nodes, gnn_dim]; populated in `precompute_frozen_cache`
        # Cache is after layer 5 (before mps.6)
        self.register_buffer(
            "frozen_cache",
            torch.zeros(n_nodes, gnn_dim),
        )
        self._cache_ready = False

        # Fallback embedding for OOV genes (not in STRING_GNN node set)
        self.fallback_emb = nn.Parameter(torch.randn(gnn_dim) * 0.02)

        # ── Deep Residual Bilinear Head ───────────────────────────────────
        # Input projection: gnn_dim → head_dim
        self.input_proj = nn.Sequential(
            nn.Linear(gnn_dim, head_dim, bias=True),
            nn.GELU(),
        )

        # 6 Residual blocks (restored from 4)
        self.resblocks = nn.ModuleList([
            ResidualBlock(head_dim, head_expand, dropout)
            for _ in range(n_resblocks)
        ])

        # Bilinear projection: head_dim → n_classes * head_dim
        self.bilinear_proj = nn.Linear(head_dim, n_classes * head_dim, bias=False)

        # Output gene embedding table [n_genes_out, head_dim]
        self.out_gene_emb = nn.Embedding(n_genes_out, head_dim)
        nn.init.xavier_uniform_(self.out_gene_emb.weight)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Node4-1-2-1-1] Trainable params: {n_trainable:,}")

    @torch.no_grad()
    def precompute_frozen_cache(self):
        """
        Run the frozen portion of STRING_GNN (emb → mps.0-5) and
        store the output in the frozen_cache buffer. Called during setup
        (before the model is moved to GPU), so all operations run on CPU.

        The STRING_GNN forward is:
            x = emb.weight
            for mp in mps:
                x = mp(x, edge_index, edge_weight) + x  ← residual outside the layer
            out = post_mp(x)

        We replicate this for layers 0-5 and cache x after layer 5.
        The frozen_cache buffer will be automatically moved to GPU by Lightning.
        """
        device = self.edge_index.device  # typically cpu at setup time
        ew     = self.edge_weight.to(device) if self.edge_weight is not None else None

        # Start from the embedding table
        x = self.gnn.emb.weight.to(device).float()  # [N, 256]

        # Run GNN layers 0 through 5 (frozen, no grad)
        # Each layer: x = layer(x, edge_index, edge_weight) + x
        ei = self.edge_index.to(device)
        for i in range(self.n_frozen):  # 0..5
            layer = self.gnn.mps[i].to(device)
            x = layer(x, ei, edge_weight=ew) + x

        self.frozen_cache.copy_(x)
        self._cache_ready = True
        print(f"[Node4-1-2-1-1] Frozen cache precomputed on {device}: "
              f"shape={self.frozen_cache.shape} (after layer {self.n_frozen - 1})")

    def _run_trainable_gnn(self) -> torch.Tensor:
        """
        Run mps.6 + mps.7 + post_mp on the frozen cache.
        Returns: [N, gnn_dim] float32 node embeddings.

        Mirrors the STRING_GNN forward pattern:
            x = layer(x, edge_index, edge_weight) + x   (residual outside layer)
            out = post_mp(x)

        After setup + Lightning GPU transfer, frozen_cache, edge_index, and
        edge_weight are all on the same GPU device.
        """
        x = self.frozen_cache  # [N, 256], on GPU, float32

        # mps.6 (trainable) — residual is added outside the GNNLayer
        x = self.gnn.mps[6](x, self.edge_index, edge_weight=self.edge_weight) + x

        # mps.7 (trainable) — residual is added outside the GNNLayer
        x = self.gnn.mps[7](x, self.edge_index, edge_weight=self.edge_weight) + x

        # post_mp (trainable Linear)
        x = self.gnn.post_mp(x)  # [N, 256]
        return x.float()          # ensure float32 output

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_indices: [B] long – STRING_GNN node indices (-1 for unknown)
        Returns:
            logits: [B, 3, 6640]
        """
        # Get final node embeddings [N, gnn_dim]
        all_embs = self._run_trainable_gnn()

        # Index embedding for each sample in batch
        # For OOV genes (index=-1), use fallback
        safe_idxs = node_indices.clamp(min=0)                  # [B]
        pert_emb  = all_embs[safe_idxs].float()                 # [B, 256]
        known_mask = node_indices >= 0
        if not known_mask.all():
            fb = self.fallback_emb.float().unsqueeze(0).expand_as(pert_emb)
            pert_emb = torch.where(
                known_mask.unsqueeze(-1).expand_as(pert_emb),
                pert_emb, fb,
            )

        # Head forward
        h = self.input_proj(pert_emb)                           # [B, 512]
        for block in self.resblocks:
            h = block(h)                                        # [B, 512]

        # Bilinear projection
        proj = self.bilinear_proj(h)                            # [B, 3*512]
        proj = proj.view(-1, N_CLASSES, self.head_dim)          # [B, 3, 512]

        # Output gene interaction
        out_embs = self.out_gene_emb.weight                     # [6640, 512]
        logits   = torch.matmul(proj, out_embs.T)               # [B, 3, 6640]
        return logits


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))
    pad    = max_sz - local_p.shape[0]
    p = local_p.to(device);  l = local_l.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], 0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], 0)
    gp = [torch.zeros_like(p) for _ in range(world_size)]
    gl = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(gp, p);  dist.all_gather(gl, l)
    rp = torch.cat([gp[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    rl = torch.cat([gl[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    return rp, rl


# ─── Quality-Filtered SWA ─────────────────────────────────────────────────────

def load_swa_predictions(
    checkpoint_dir: Path,
    datamodule: pl.LightningDataModule,
    base_lit_module: "StringGNNPartialLitModule",
    top_k: int = 15,
    temperature: float = 3.0,
    val_f1_threshold: float = 0.490,
    device: str = "cuda:0",
) -> Optional[Tuple[np.ndarray, List[str], List[str]]]:
    """
    Post-hoc quality-filtered exponentially-weighted SWA.

    Scans checkpoint_dir for periodic checkpoints with val_f1 in the filename,
    selects top-k by val_f1, computes test predictions for each, and combines
    them via softmax-temperature weighted average.

    Returns (probs_avg, pert_ids, symbols) or None if fewer than 2 valid checkpoints.
    """
    import re
    import copy

    # Find all periodic checkpoints with val_f1 in name
    ckpt_files = list(checkpoint_dir.glob("periodic-*.ckpt"))
    if not ckpt_files:
        print("[SWA] No periodic checkpoints found, skipping SWA.")
        return None

    # Parse val_f1 from filename using regex
    pattern = re.compile(r'val_f1=([0-9]+\.[0-9]+)')
    ckpt_with_scores: List[Tuple[float, Path]] = []
    for ckpt in ckpt_files:
        m = pattern.search(ckpt.name)
        if m:
            score = float(m.group(1))
            if score >= val_f1_threshold:
                ckpt_with_scores.append((score, ckpt))

    if len(ckpt_with_scores) < 2:
        print(f"[SWA] Only {len(ckpt_with_scores)} checkpoints meet threshold "
              f"{val_f1_threshold}, skipping SWA.")
        return None

    # Sort by val_f1 descending, take top_k
    ckpt_with_scores.sort(key=lambda x: x[0], reverse=True)
    selected = ckpt_with_scores[:top_k]
    print(f"[SWA] Selected {len(selected)}/{len(ckpt_with_scores)} qualifying checkpoints "
          f"(top-{top_k}, temp={temperature}, threshold={val_f1_threshold})")
    for score, path in selected:
        print(f"  val_f1={score:.4f}: {path.name}")

    # Compute temperature-weighted softmax weights
    scores_arr = np.array([s for s, _ in selected])
    # Scale scores relative to max for numerical stability
    scaled = (scores_arr - scores_arr.max()) * temperature
    weights = np.exp(scaled)
    weights = weights / weights.sum()
    print(f"[SWA] Weights range: [{weights.min():.3f}, {weights.max():.3f}]")

    # Collect predictions from each checkpoint
    all_probs: List[np.ndarray] = []
    pert_ids_out: Optional[List[str]] = None
    symbols_out: Optional[List[str]] = None

    test_loader = datamodule.test_dataloader()

    for i, (score, ckpt_path) in enumerate(selected):
        # Load a fresh copy of the module with this checkpoint
        try:
            lit_copy = StringGNNPartialLitModule.load_from_checkpoint(
                str(ckpt_path),
                map_location=device,
            )
        except Exception as e:
            print(f"[SWA] Warning: could not load {ckpt_path.name}: {e}")
            continue

        lit_copy = lit_copy.to(device)
        lit_copy.eval()

        # Manually recreate model (setup with datamodule info)
        if not hasattr(lit_copy, 'model') or lit_copy.model is None:
            lit_copy._edge_index  = datamodule.edge_index
            lit_copy._edge_weight = datamodule.edge_weight
            lit_copy._n_nodes     = datamodule.n_nodes
            lit_copy.model = PartialGNNWithHead(
                edge_index  = datamodule.edge_index,
                edge_weight = datamodule.edge_weight,
                n_nodes     = datamodule.n_nodes,
                dropout     = lit_copy.hparams.dropout,
            )
            lit_copy.model.precompute_frozen_cache()

            # Load state dict
            state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            if "state_dict" in state:
                state = state["state_dict"]
            lit_copy.load_state_dict(state, strict=False)

        lit_copy = lit_copy.to(device)
        lit_copy.model = lit_copy.model.to(device)
        lit_copy.eval()

        batch_probs: List[torch.Tensor] = []
        batch_pids: List[str] = []
        batch_syms: List[str] = []

        with torch.no_grad():
            for batch in test_loader:
                node_idx = batch["node_index"].to(device)
                logits = lit_copy.model(node_idx)
                probs  = torch.softmax(logits, dim=1).cpu().float()
                batch_probs.append(probs)
                batch_pids.extend(batch["pert_id"])
                batch_syms.extend(batch["symbol"])

        ckpt_probs = torch.cat(batch_probs, 0).numpy()  # [N, 3, 6640]
        all_probs.append(ckpt_probs)

        if pert_ids_out is None:
            pert_ids_out = batch_pids
            symbols_out  = batch_syms

        print(f"[SWA] [{i+1}/{len(selected)}] Loaded ckpt val_f1={score:.4f}, "
              f"probs shape={ckpt_probs.shape}")
        del lit_copy

    if len(all_probs) < 2:
        print("[SWA] Too few valid checkpoints loaded, skipping SWA.")
        return None

    # Adjust weights to match loaded count
    final_weights = weights[:len(all_probs)]
    final_weights = final_weights / final_weights.sum()

    # Weighted average
    probs_avg = np.zeros_like(all_probs[0])
    for w, p in zip(final_weights, all_probs):
        probs_avg += w * p

    print(f"[SWA] Averaged {len(all_probs)} checkpoints. "
          f"Probs shape: {probs_avg.shape}")
    return probs_avg, pert_ids_out, symbols_out


# ─── LightningModule ──────────────────────────────────────────────────────────

class StringGNNPartialLitModule(pl.LightningModule):

    def __init__(
        self,
        backbone_lr:     float = 1e-5,
        head_lr:         float = 5e-4,
        muon_lr:         float = 0.005,
        weight_decay:    float = 1e-3,
        focal_gamma:     float = 2.0,
        class_weights:   List[float] = None,
        dropout:         float = 0.40,
        warmup_t0:       int   = 300,   # Corrected T_0: ~13 epochs/cycle for 2-GPU DDP
        label_smoothing: float = 0.02,  # Light smoothing (reduced from 0.05)
    ):
        super().__init__()
        if class_weights is None:
            class_weights = [2.0, 0.5, 4.0]
        self.save_hyperparameters()

        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str]          = []
        self._test_symbols:  List[str]          = []
        self._test_labels:   List[torch.Tensor] = []

        self._edge_index:  Optional[torch.Tensor] = None
        self._edge_weight: Optional[torch.Tensor] = None
        self._n_nodes:     int                    = 18870

    def setup(self, stage=None):
        dm = self.trainer.datamodule if self.trainer is not None else None
        if dm is not None and hasattr(dm, "edge_index"):
            self._edge_index  = dm.edge_index
            self._edge_weight = dm.edge_weight
            self._n_nodes     = dm.n_nodes

        self.model = PartialGNNWithHead(
            edge_index  = self._edge_index,
            edge_weight = self._edge_weight,
            n_nodes     = self._n_nodes,
            dropout     = self.hparams.dropout,
        )

        # Precompute frozen GNN cache on CPU during setup.
        # The buffer will automatically be moved to GPU when the trainer
        # calls .to(device) on the module.
        self.model.precompute_frozen_cache()

        # Cast trainable parameters to float32 for stable optimization
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Class weights buffer
        cw = torch.tensor(self.hparams.class_weights, dtype=torch.float32)
        self.register_buffer("_class_weights", cw)

    def forward(self, node_indices):
        return self.model(node_indices)

    def _loss(self, logits, labels):
        # Reshape: [B, 3, G] → [B*G, 3], [B, G] → [B*G]
        logits_2d = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)
        labels_1d = labels.reshape(-1)
        return focal_cross_entropy(
            logits_2d, labels_1d,
            gamma=self.hparams.focal_gamma,
            class_weights=self._class_weights.to(logits.device),
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
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
        self._val_preds.clear();  self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
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

        # Deduplicate by pert_id (DDP DistributedSampler may pad)
        seen_pids: set = set()
        keep_indices:  List[int] = []
        for i, pid in enumerate(all_pert):
            if pid not in seen_pids:
                seen_pids.add(pid)
                keep_indices.append(i)
        if len(keep_indices) < len(all_pert):
            self.print(f"[Node4-1-2-1-1] Deduplicating: {len(all_pert)} → {len(keep_indices)}")
            all_probs  = all_probs[keep_indices]
            all_labels = all_labels[keep_indices]
            all_pert   = [all_pert[i] for i in keep_indices]
            all_syms   = [all_syms[i]  for i in keep_indices]

        if self.trainer.is_global_zero:
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for pid, sym, probs in zip(all_pert, all_syms, all_probs.numpy()):
                    fh.write(f"{pid}\t{sym}\t{json.dumps(probs.tolist())}\n")
            self.print(f"[Node4-1-2-1-1] Saved test predictions → {pred_path}")
            if all_labels.any():
                f1 = compute_per_gene_f1(all_probs.numpy(), all_labels.numpy())
                self.print(f"[Node4-1-2-1-1] Self-computed test F1 (best ckpt) = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams
        model = self.model

        # ── Parameter group separation ───────────────────────────────────
        # Group 1 (Muon): 2D weight matrices inside ResidualBlocks (fc1, fc2)
        # Group 2 (AdamW): head scalars/biases/embeddings/LayerNorms
        # Group 3 (AdamW): backbone mps.6+7 and post_mp
        resblock_2d_params = []
        other_head_params  = []
        backbone_params    = []

        # Backbone trainable params: mps.6, mps.7 and post_mp
        backbone_param_ids = set()
        for p in model.gnn.mps[6].parameters():
            if p.requires_grad:
                backbone_params.append(p)
                backbone_param_ids.add(id(p))
        for p in model.gnn.mps[7].parameters():
            if p.requires_grad:
                backbone_params.append(p)
                backbone_param_ids.add(id(p))
        for p in model.gnn.post_mp.parameters():
            if p.requires_grad:
                backbone_params.append(p)
                backbone_param_ids.add(id(p))

        # Head params: separate Muon-eligible 2D matrices from others
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in backbone_param_ids:
                continue
            # ResidualBlock fc1, fc2 weights are 2D and suitable for Muon
            is_resblock_weight = (
                "resblocks" in name
                and ("fc1.weight" in name or "fc2.weight" in name)
            )
            # bilinear_proj.weight is also a 2D matrix, include for Muon
            is_bilinear_weight = ("bilinear_proj.weight" in name)
            if is_resblock_weight or is_bilinear_weight:
                resblock_2d_params.append(p)
            else:
                other_head_params.append(p)

        self.print(f"[Node4-1-2-1-1] Optimizer groups: "
                   f"Muon={len(resblock_2d_params)} tensors, "
                   f"AdamW_head={len(other_head_params)} tensors, "
                   f"AdamW_backbone={len(backbone_params)} tensors")

        # ── MuonWithAuxAdam ──────────────────────────────────────────────
        try:
            from muon import MuonWithAuxAdam
            param_groups = [
                dict(
                    params=resblock_2d_params,
                    use_muon=True,
                    lr=hp.muon_lr,
                    momentum=0.95,
                    weight_decay=1e-4,  # Mild L2 regularization for ResBlock matrices
                ),
                dict(
                    params=other_head_params,
                    use_muon=False,
                    lr=hp.head_lr,
                    betas=(0.9, 0.95),
                    weight_decay=hp.weight_decay,
                ),
                dict(
                    params=backbone_params,
                    use_muon=False,
                    lr=hp.backbone_lr,
                    betas=(0.9, 0.95),
                    weight_decay=0.0,  # No weight decay for backbone
                ),
            ]
            optimizer = MuonWithAuxAdam(param_groups)
            self.print("[Node4-1-2-1-1] Using MuonWithAuxAdam optimizer.")
        except ImportError:
            # Fallback to AdamW if Muon is not available
            self.print("[Node4-1-2-1-1] Muon not available, falling back to AdamW.")
            all_params_groups = [
                {"params": resblock_2d_params + other_head_params,
                 "lr": hp.head_lr, "weight_decay": hp.weight_decay},
                {"params": backbone_params,
                 "lr": hp.backbone_lr, "weight_decay": 0.0},
            ]
            optimizer = torch.optim.AdamW(all_params_groups)

        # ── CosineAnnealingWarmRestarts ──────────────────────────────────
        # T_0=300 steps (corrected for 2-GPU DDP: ~13 epochs/cycle)
        # With 2 GPUs: steps/epoch = ceil(1416 / (4 * 2 * 8)) ≈ 23
        # T_0=300 → ~13 epochs/cycle (validated in parent node)
        # The scheduler is updated per step (interval="step")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hp.warmup_t0,
            T_mult=1,
            eta_min=1e-7,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # Update every optimizer step
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
    p = argparse.ArgumentParser(
        description="Node 4-1-2-1-1 – STRING_GNN Partial FT (mps.6+7+post_mp) + Rank-512 Head (6 ResBlocks) + Muon + Warm Restarts + SWA"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--backbone-lr",        type=float, default=1e-5)
    p.add_argument("--head-lr",            type=float, default=5e-4)
    p.add_argument("--muon-lr",            type=float, default=0.005)
    p.add_argument("--weight-decay",       type=float, default=1e-3)
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--dropout",            type=float, default=0.40)
    p.add_argument("--warmup-t0",          type=int,   default=300)
    p.add_argument("--label-smoothing",    type=float, default=0.02)
    p.add_argument("--micro-batch-size",   type=int,   default=4)
    p.add_argument("--global-batch-size",  type=int,   default=64)
    p.add_argument("--max-epochs",         type=int,   default=200)
    p.add_argument("--patience",           type=int,   default=50)
    p.add_argument("--periodic-ckpt-freq", type=int,   default=5,
                   help="Save periodic checkpoints every N epochs for SWA pool")
    p.add_argument("--swa-top-k",          type=int,   default=15,
                   help="Number of top checkpoints to use in SWA averaging")
    p.add_argument("--swa-temperature",    type=float, default=3.0,
                   help="Temperature for softmax weighting of SWA checkpoints")
    p.add_argument("--swa-threshold",      type=float, default=0.490,
                   help="Minimum val_f1 for SWA checkpoint inclusion")
    p.add_argument("--num-workers",        type=int,   default=4)
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

    dm  = StringGNNDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = StringGNNPartialLitModule(
        backbone_lr    = args.backbone_lr,
        head_lr        = args.head_lr,
        muon_lr        = args.muon_lr,
        weight_decay   = args.weight_decay,
        focal_gamma    = args.focal_gamma,
        class_weights  = [2.0, 0.5, 4.0],
        dropout        = args.dropout,
        warmup_t0      = args.warmup_t0,
        label_smoothing= args.label_smoothing,
    )

    # Best checkpoint monitor (used for test with ckpt_path='best')
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )

    # Periodic checkpoints for SWA pool (every N epochs, keep all)
    periodic_ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="periodic-epoch={epoch:04d}-val_f1={val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=-1,    # Keep all periodic checkpoints
        every_n_epochs=args.periodic_ckpt_freq,
        save_last=False,
    )

    es_cb  = EarlyStopping(monitor="val_f1", mode="max",
                           patience=args.patience, min_delta=1e-5)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)
    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps    = -1
    limit_train: float | int = 1.0
    limit_val:   float | int = 1.0
    limit_test:  float | int = 1.0
    fast_dev_run = False
    if args.debug_max_step is not None:
        max_steps   = args.debug_max_step;  limit_train = args.debug_max_step
        limit_val   = 2;  limit_test = 2
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
        callbacks=[ckpt_cb, periodic_ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)

    # ── Phase 1: Test with best single checkpoint ─────────────────────────────
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    # ── Phase 2: Quality-filtered SWA (rank 0 only, after training) ───────────
    # Skip SWA in debug mode or fast_dev_run
    if (args.debug_max_step is None and not args.fast_dev_run
            and trainer.is_global_zero):

        ckpt_dir = out_dir / "checkpoints"
        device_str = f"cuda:{trainer.local_rank}" if torch.cuda.is_available() else "cpu"

        print(f"\n[SWA] Starting quality-filtered SWA (top_k={args.swa_top_k}, "
              f"temp={args.swa_temperature}, threshold={args.swa_threshold})...")

        # Need a fresh datamodule on the same device for SWA inference
        swa_dm = StringGNNDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
        swa_dm.setup()

        swa_result = load_swa_predictions(
            checkpoint_dir   = ckpt_dir,
            datamodule       = swa_dm,
            base_lit_module  = lit,
            top_k            = args.swa_top_k,
            temperature      = args.swa_temperature,
            val_f1_threshold = args.swa_threshold,
            device           = device_str,
        )

        if swa_result is not None:
            swa_probs, swa_pids, swa_syms = swa_result

            # Save SWA predictions (overwrite the best-ckpt predictions)
            pred_path_swa = out_dir / "test_predictions.tsv"
            with open(pred_path_swa, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for pid, sym, probs in zip(swa_pids, swa_syms, swa_probs):
                    fh.write(f"{pid}\t{sym}\t{json.dumps(probs.tolist())}\n")
            print(f"[SWA] Saved SWA predictions → {pred_path_swa}")

            # Self-evaluate SWA predictions against test labels (if available)
            # Load test labels for self-eval
            try:
                test_df = pd.read_csv(Path(args.data_dir) / "test.tsv", sep="\t")
                if "label" in test_df.columns:
                    pid_to_labels: Dict[str, List[int]] = {}
                    for _, row in test_df.iterrows():
                        pid_to_labels[row["pert_id"]] = [x + 1 for x in json.loads(row["label"])]
                    ordered_labels = [pid_to_labels[pid] for pid in swa_pids if pid in pid_to_labels]
                    if len(ordered_labels) == len(swa_pids):
                        labels_np = np.array(ordered_labels, dtype=np.int64)
                        f1_swa = compute_per_gene_f1(swa_probs, labels_np)
                        print(f"[SWA] Self-computed test F1 (SWA ensemble) = {f1_swa:.4f}")
            except Exception as e:
                print(f"[SWA] Could not compute SWA F1 self-eval: {e}")
        else:
            print("[SWA] SWA not applied; using best single-checkpoint predictions.")

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 4-1-2-1-1 – STRING_GNN Partial FT (mps.6+7+post_mp) + Rank-512 Head (6 ResBlocks) + Muon + Warm Restarts + SWA\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
