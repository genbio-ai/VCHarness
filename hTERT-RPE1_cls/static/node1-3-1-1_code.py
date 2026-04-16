"""
Node 1-3-1-1: Frozen STRING_GNN + Wider Bilinear Head (rank=512, hidden=768)

Architecture:
  - STRING_GNN (FULLY FROZEN - proven optimal from node1-2)
    Pre-computed frozen embeddings extracted once at setup and stored as buffer.
    No conditioning modules — ALL prior conditioning attempts consistently degraded performance.
  - Wider bilinear interaction: rank increased from 256 to 512
    Linear(768 → 3*512), out_gene_emb [6640, 512] instead of [6640, 256]
    Doubles the expressiveness of the gene-gene correlation structure.
  - Wider MLP head: hidden_dim=768 (up from 512)
    More representational capacity without touching frozen PPI embeddings.
  - Dropout=0.25 (up from 0.2 to compensate for larger model)
  - Focal loss (gamma=2.0) — NO label smoothing (node1-2-2 showed it hurt F1)
  - Calibrated cosine LR: total_steps = effective_steps_per_epoch * 120
    Calibrated to node1-3-1's training pattern (best at epoch 89, ~140 total)
  - Patience=50 (captures secondary improvement window, proven in node1-3)

Key Changes from Parent (node1-3-1, F1=0.4714):
  1. REMOVE transductive pert_U/pert_V conditioning (proven to degrade val/test performance)
     - Feedback root cause: pert_U rows for val/test genes never updated → adds noise
     - Confirmed: conditioning removed → should restore to node1-2 baseline (F1=0.4912)
  2. INCREASE bilinear rank from 256 to 512
     - Doubles expressiveness of bilinear gene-gene interaction
     - Low risk: doesn't touch frozen PPI embeddings
     - Expected: +0.5-2% F1 from richer gene correlation modeling
  3. INCREASE hidden_dim from 512 to 768
     - Wider MLP provides more capacity to process frozen PPI features
     - Balanced with slightly higher dropout (0.25) to control overfitting
  4. CALIBRATE total_steps to 120 epochs (not 150)
     - node1-3-1 best at epoch 89; node1-2 best at epoch 98
     - 120 epochs is a middle estimate for well-calibrated cosine annealing
  5. KEEP patience=50 (validated in node1-3 to enable secondary improvement window)

Differentiation from siblings:
  - node1-2 (F1=0.4912): hidden=512, rank=256, no conditioning (our baseline)
  - node1-2-2 (F1=0.4664): InductiveCondMLP + label smoothing (failed — conditioning harmful)
  - node1-3-1 (parent, F1=0.4714): Transductive pert_U/V (failed — not inductive)
  - This node: pure frozen backbone + WIDER bilinear (rank=512) + WIDER MLP (hidden=768)

Hypothesis: The frozen PPI embeddings are optimal as-is (proven in node1-2).
The bottleneck is the bilinear head's expressiveness. Increasing rank from 256 to 512
doubles the gene-gene correlation space, potentially capturing interactions not encoded
at rank=256, while preserving the clean frozen embedding foundation.

Expected Performance: F1=0.49–0.52
  - Floor: node1-2's F1=0.4912 (same frozen PPI foundation + bilinear structure)
  - Upside: wider bilinear rank captures richer gene-gene correlations
  - Risk: larger head may overfit slightly on 1,416 training samples (mitigated by dropout=0.25)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

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


# ─── STRING_GNN Loading ───────────────────────────────────────────────────────

def load_frozen_string_gnn_embeddings(
    model_dir: Path,
    device: torch.device,
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """Load STRING_GNN model, extract frozen embeddings, then delete model.

    Returns:
        embeddings: [N_nodes, 256] numpy array of frozen pretrained embeddings
        node_names: list of Ensembl gene IDs (aligned with node indices)
        node_name_to_idx: dict mapping Ensembl ID -> node index
    """
    node_names = json.loads((model_dir / "node_names.json").read_text())
    graph = torch.load(model_dir / "graph_data.pt", weights_only=False)
    node_name_to_idx = {name: i for i, name in enumerate(node_names)}

    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    model = model.to(device).eval()

    # Freeze ALL parameters — fully frozen backbone
    for p in model.parameters():
        p.requires_grad = False

    edge_index = graph["edge_index"].to(device)
    edge_weight = graph.get("edge_weight")
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    # Extract embeddings in frozen mode
    with torch.no_grad():
        outputs = model(
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
    embeddings = outputs.last_hidden_state.float().cpu().numpy()

    # Delete model to free GPU memory — we only need the embeddings
    del model
    del edge_index
    del edge_weight
    torch.cuda.empty_cache()

    return embeddings, node_names, node_name_to_idx


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset.

    Each sample provides:
    - pert_id: Ensembl gene ID
    - symbol: gene symbol
    - node_idx: STRING_GNN node index (for embedding lookup), -1 if OOV
    - label: [N_GENES_OUT] integer class labels 0/1/2
    """

    def __init__(
        self,
        df: pd.DataFrame,
        node_name_to_idx: Dict[str, int],
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        # Map pert_id to STRING_GNN node index (-1 for OOV)
        node_indices = []
        for pert_id in self.pert_ids:
            if pert_id in node_name_to_idx:
                node_indices.append(node_name_to_idx[pert_id])
            else:
                node_indices.append(-1)
        self.node_indices = torch.tensor(node_indices, dtype=torch.long)  # [N]

        self.has_labels = has_labels
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
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "node_idx": self.node_indices[idx],  # long scalar, -1 if OOV
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule using STRING_GNN node indices (precomputed frozen embeddings).

    Since the backbone is fully frozen and embeddings are precomputed at setup,
    we pass node indices to the model which does a simple embedding lookup.
    This avoids running STRING_GNN forward pass per batch — much faster training.
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

        # Load node names for dataset construction
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        self.node_name_to_idx = {name: i for i, name in enumerate(node_names)}
        self.n_nodes = len(node_names)

        # Load all splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        # Coverage report
        train_in_vocab = sum(p in self.node_name_to_idx for p in dfs['train']['pert_id'])
        print(f"[DataModule] Coverage: {train_in_vocab}/{len(dfs['train'])} train genes in STRING_GNN")

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


# ─── Model ────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual MLP block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout + skip."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.25):
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


class FrozenGNNWiderBilinearHead(nn.Module):
    """Prediction head with frozen STRING_GNN embeddings + wider bilinear interaction.

    Key design differences from parent (node1-3-1):
    - NO conditioning modules (all prior conditioning consistently degraded performance)
    - Wider bilinear rank: 512 vs. 256 (doubles gene-gene correlation expressiveness)
    - Wider MLP hidden dim: 768 vs. 512 (more processing capacity)
    - Slightly higher dropout: 0.25 vs. 0.2 (compensates for larger model)

    The frozen PPI embeddings are proven optimal as-is. This node focuses on
    improving head expressiveness rather than modifying the embedding pathway.
    """

    def __init__(
        self,
        frozen_embeddings: torch.Tensor,  # [N_nodes, gnn_dim] — precomputed
        gnn_dim: int = 256,
        hidden_dim: int = 768,          # WIDER: 768 vs. 512 in parent
        bilinear_rank: int = 512,       # WIDER: 512 vs. 256 in parent
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        dropout: float = 0.25,          # Slightly higher to compensate for larger model
        n_residual_layers: int = 6,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_genes_out = n_genes_out
        self.bilinear_rank = bilinear_rank
        self.gnn_dim = gnn_dim

        # Frozen embeddings stored as a buffer (not trainable)
        # Shape: [N_nodes, gnn_dim]
        self.register_buffer("frozen_emb", frozen_embeddings.float())

        # OOV fallback embedding for genes not in STRING_GNN (~6.4% of dataset)
        self.oov_emb = nn.Embedding(1, gnn_dim)

        # Input normalization
        self.input_norm = nn.LayerNorm(gnn_dim)

        # Projection: gnn_dim -> hidden_dim
        self.proj_in = nn.Linear(gnn_dim, hidden_dim)

        # Deep residual MLP — wider than parent
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, expand=4, dropout=dropout)
             for _ in range(n_residual_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear projection: hidden_dim -> n_classes * bilinear_rank (WIDER)
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * bilinear_rank)

        # Head dropout
        self.head_dropout = nn.Dropout(dropout)

        # Output gene embeddings [n_genes_out, bilinear_rank] — WIDER
        # RANDOM initialization (std=0.02) — same proven approach as node1-2
        self.out_gene_emb = nn.Parameter(torch.empty(n_genes_out, bilinear_rank))
        nn.init.normal_(self.out_gene_emb, std=0.02)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.oov_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)

    def forward(
        self,
        node_idx: torch.Tensor,  # [B] long, -1 for OOV
    ) -> torch.Tensor:
        """
        Args:
            node_idx: [B] long - STRING_GNN node indices, -1 if OOV
        Returns:
            logits: [B, 3, 6640]
        """
        B = node_idx.shape[0]
        device = node_idx.device

        # --- Step 1: Frozen embedding lookup (O(1) per batch, no backbone forward) ---
        in_vocab_mask = (node_idx >= 0)  # [B] bool
        # Clamp OOV to 0 for safe indexing (will be replaced by oov_emb)
        safe_idx = node_idx.clamp(min=0)  # [B]

        # Gather frozen node embeddings for in-vocab genes
        gnn_emb = self.frozen_emb[safe_idx]  # [B, gnn_dim]

        # Replace OOV embeddings with learned fallback
        oov_token = self.oov_emb(torch.zeros(B, dtype=torch.long, device=device))  # [B, gnn_dim]
        in_vocab_f = in_vocab_mask.unsqueeze(1).float()  # [B, 1]
        gnn_emb = gnn_emb * in_vocab_f + oov_token * (1.0 - in_vocab_f)  # [B, gnn_dim]

        # --- Step 2: MLP head ---
        x = self.input_norm(gnn_emb)   # [B, gnn_dim]
        x = self.proj_in(x)            # [B, hidden_dim]

        for blk in self.res_blocks:
            x = blk(x)
        x = self.norm_out(x)           # [B, hidden_dim]

        # --- Step 3: Bilinear interaction (WIDER rank) ---
        x = self.head_dropout(x)
        pert_proj = self.proj_bilinear(x)                          # [B, n_classes * bilinear_rank]
        pert_proj = pert_proj.view(B, self.n_classes, self.bilinear_rank)  # [B, 3, bilinear_rank]

        # Bilinear: [B, 3, bilinear_rank] x [bilinear_rank, n_genes_out] → [B, 3, n_genes_out]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)  # [B, 3, 6640]

        return logits


# ─── Focal Loss ───────────────────────────────────────────────────────────────

def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss: -(1-p_t)^gamma * log(p_t). No class weights (focal handles imbalance).
    No label smoothing — node1-2-2 demonstrated label smoothing hurts F1 despite improving calibration.

    Args:
        logits: [B, 3, G] raw logits
        labels: [B, G] integer class labels (0, 1, 2)
        gamma: focusing parameter
    Returns:
        Scalar loss value.
    """
    B, C, G = logits.shape

    # Reshape to [B*G, 3] and [B*G]
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
    labels_flat = labels.reshape(-1)                        # [B*G]

    # Cross-entropy per element (no label smoothing)
    ce_loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        reduction="none",
    )  # [B*G]

    # Focal weighting
    with torch.no_grad():
        probs = F.softmax(logits_flat, dim=1)           # [B*G, 3]
        pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)  # [B*G]
        focal_weight = (1.0 - pt).pow(gamma)            # [B*G]

    loss = (focal_weight * ce_loss).mean()
    return loss


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

    real_preds  = torch.cat([g_preds[i][: all_sizes[i].item()].cpu()  for i in range(world_size)], 0)
    real_labels = torch.cat([g_labels[i][: all_sizes[i].item()].cpu() for i in range(world_size)], 0)
    return real_preds, real_labels


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbationLitModule(pl.LightningModule):
    """LightningModule for gene-perturbation DEG prediction (Node 1-3-1-1).

    Key design: frozen STRING_GNN embeddings (precomputed) + wider bilinear head.
    No backbone forward pass during training — only simple embedding table lookup.
    No conditioning modules — all conditioning attempts consistently degraded performance.
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 768,
        bilinear_rank: int = 512,
        n_residual_layers: int = 6,
        dropout: float = 0.25,
        lr: float = 5e-4,
        weight_decay: float = 1e-3,
        focal_gamma: float = 2.0,
        warmup_steps: int = 50,
        total_steps: int = 1000,
        n_nodes: int = 18870,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Accumulation buffers
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams
        device = self.device if self.device.type != "meta" else torch.device("cpu")

        # Load STRING_GNN, extract frozen embeddings, then delete model
        # This is more memory-efficient than keeping the backbone during training
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            print(f"[Node1-3-1-1] Loading STRING_GNN frozen embeddings on device={device}")
        frozen_emb_np, _, node_name_to_idx = load_frozen_string_gnn_embeddings(
            STRING_GNN_DIR, device
        )
        frozen_emb_tensor = torch.from_numpy(frozen_emb_np).float()

        self.model = FrozenGNNWiderBilinearHead(
            frozen_embeddings=frozen_emb_tensor,
            gnn_dim=hp.gnn_dim,
            hidden_dim=hp.hidden_dim,
            bilinear_rank=hp.bilinear_rank,
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            dropout=hp.dropout,
            n_residual_layers=hp.n_residual_layers,
        )

        # Ensure float32 for all trainable parameters
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

    def forward(self, node_idx: torch.Tensor) -> torch.Tensor:
        return self.model(node_idx)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return focal_loss(logits, labels, gamma=self.hparams.focal_gamma)

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["node_idx"])
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
        logits = self(batch["node_idx"])
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
            self.print(f"[Node1-3-1-1] Saved test predictions → {pred_path} ({len(seen_ids)} unique samples)")

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-3-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Single optimizer group — no backbone params (backbone is frozen and deleted)
        # Only head params are trainable: oov_emb, proj_in, res_blocks, proj_bilinear, out_gene_emb
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=hp.lr,
            weight_decay=hp.weight_decay,
        )

        # Cosine annealing with linear warmup
        # Calibrated to node1-3-1's training pattern: best at epoch 89, stopped at 140
        # Using 120 epochs as target calibration (intermediate between 89 and 150)
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

    # ── Checkpoint: save only trainable params ─────────────────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_sd = {}
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        for k, v in full_sd.items():
            if k in trainable_keys or k in buffer_keys:
                trainable_sd[k] = v
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving ckpt: {trained}/{total} trainable params ({100*trained/total:.1f}%)")
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 1-3-1-1 – Frozen STRING_GNN + Wider Bilinear Head (rank=512, hidden=768)"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--gnn-dim",            type=int,   default=256)
    p.add_argument("--hidden-dim",         type=int,   default=768)
    p.add_argument("--bilinear-rank",      type=int,   default=512)
    p.add_argument("--n-residual-layers",  type=int,   default=6)
    p.add_argument("--dropout",            type=float, default=0.25)
    p.add_argument("--lr",                 type=float, default=5e-4)
    p.add_argument("--weight-decay",       type=float, default=1e-3)
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--warmup-steps",       type=int,   default=50)
    p.add_argument("--micro-batch-size",   type=int,   default=16)
    p.add_argument("--global-batch-size",  type=int,   default=64)
    p.add_argument("--max-epochs",         type=int,   default=300)
    p.add_argument("--patience",           type=int,   default=50)
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

    # DataModule
    dm = PerturbationDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    # Estimate total training steps for LR schedule
    # Calibrated to observed training patterns:
    #   - node1-2: best at epoch 98, stopped at epoch 128 (patience=30)
    #   - node1-3-1: best at epoch 89, stopped at epoch 139 (patience=50)
    # Using 120 epochs as calibration target (between the two best-epoch observations)
    steps_per_epoch = max(1, len(dm.train_ds) // (args.micro_batch_size * n_gpus))
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)
    expected_training_epochs = min(args.max_epochs, 120)
    total_steps = effective_steps_per_epoch * expected_training_epochs

    # LightningModule
    lit = PerturbationLitModule(
        gnn_dim=args.gnn_dim,
        hidden_dim=args.hidden_dim,
        bilinear_rank=args.bilinear_rank,
        n_residual_layers=args.n_residual_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        n_nodes=dm.n_nodes,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=1, save_last=True,
    )
    es_cb = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-5)
    lr_cb = LearningRateMonitor(logging_interval="step")
    pb_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    # Debug / fast-dev-run settings
    max_steps:           int           = -1
    limit_train_batches: float | int   = 1.0
    limit_val_batches:   float | int   = 1.0
    limit_test_batches:  float | int   = 1.0
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
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        gradient_clip_val=1.0,   # gradient clipping for stable training
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
            f"Node 1-3-1-1 – Frozen STRING_GNN + Wider Bilinear Head (rank=512, hidden=768)\n"
            f"Test results from trainer: {test_results}\n"
            f"(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
