"""
Node 1-1-3-1-1-1 – STRING_GNN Partial Fine-Tuning + Deep Residual Bilinear MLP
                    with MuonWithAuxAdam and Cosine Warm Restarts (SGDR)

This node represents a strategic pivot from the parent's AIDO.Cell + STRING_GNN hybrid
to a pure STRING_GNN-based architecture, which consistently outperforms AIDO.Cell-based
approaches by 0.05+ F1 across the MCTS search tree.

Architecture:
  Backbone: STRING_GNN (18,870 nodes, 256-dim PPI embeddings)
    - mps.0–6: frozen (precomputed as buffer for memory efficiency)
    - mps.7 + post_mp: partially fine-tuned (lr=1e-5, ~67K trainable backbone params)
  Head: 6-layer Deep Residual Bilinear MLP
    - hidden_dim=512, expand=4, rank=512, dropout=0.30
    - ~17.1M trainable head parameters
  Output: bilinear interaction [B, 3, 512] x [6640, 512] → logits [B, 3, 6640]

Optimizer: MuonWithAuxAdam (3 groups)
  - Muon lr=0.005 for ResBlock 2D hidden weight matrices
  - AdamW lr=5e-4 for head scalars, embeddings, output
  - AdamW lr=1e-5 for backbone (mps.7 + post_mp)

LR Schedule: CosineAnnealingWarmRestarts (SGDR)
  - T_0=600 steps, T_mult=1 (cycles of ~27 epochs)
  - eta_min=1e-6 (prevents LR=0 frozen-epoch pathology)
  - Ascending staircase mechanism drives successive improvement across cycles

Loss: Class-weighted focal cross-entropy
  - gamma=2.0, class_weights=[2.0, 0.5, 4.0] (down, neutral, up)
  - No label smoothing (proven by tree best node1-1-2-1-1-1-1 at 0.5035)

Key differences from parent (node1-1-3-1-1, F1=0.4858):
1. [MAJOR] Pivot: STRING_GNN primary backbone (not AIDO.Cell) — removes synthetic
   input representation limitation
2. [MAJOR] Muon optimizer for ResBlock 2D matrices — validated +0.01-0.02 F1 gain
3. [MAJOR] Deep 6-layer ResidualBlock head (hidden=512, rank=512) vs. bilinear-only
4. [MAJOR] SGDR warm restarts (T_0=600, ~27 epochs/cycle) — enables staircase
   successive improvements across cycles
5. [MODERATE] Partial backbone fine-tuning (mps.7+post_mp) vs. fully frozen
6. [MODERATE] Class weights [2.0, 0.5, 4.0] vs. [2.5, 1.0, 5.5]
7. [MINOR] Extended training: max_epochs=250, patience=80

Memory rationale:
  - node1-1-2-1-1-1-1 (F1=0.5035): STRING_GNN + partial FT + Muon + bilinear rank-512
  - node1-2-2-2-1 (F1=0.5099): same + SGDR T_0=600 warm restarts
  - All AIDO.Cell lineage nodes top out at ~0.44-0.49 vs STRING_GNN at 0.50+
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import argparse
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# ─── Constants ────────────────────────────────────────────────────────────────
N_GENES_OUT = 6640
N_CLASSES = 3
STRING_GNN_DIR = "/home/Models/STRING_GNN"
GNN_DIM = 256

# Class weights: down=2.0, neutral=0.5, up=4.0
# Proven in tree best STRING_GNN nodes (node1-1-2-1-1-1-1, node1-2-2-2-1, etc.)
CLASS_WEIGHTS = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float32)


# ─── Metric ───────────────────────────────────────────────────────────────────
def compute_per_gene_f1(pred_logits_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Mean per-gene macro-F1 (matches data/calc_metric.py _evaluate_deg)."""
    pred_classes = pred_logits_np.argmax(axis=1)  # [N, G]
    n_genes = labels_np.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels_np[:, g]
        yh = pred_classes[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Loss ─────────────────────────────────────────────────────────────────────
def focal_cross_entropy(logits, targets, class_weights, gamma=2.0, label_smoothing=0.0):
    """Focal cross-entropy with class weights and optional label smoothing."""
    ce = F.cross_entropy(
        logits, targets,
        weight=class_weights.to(logits.device),
        reduction="none",
        label_smoothing=label_smoothing,
    )
    pt = torch.exp(-ce)
    focal = (1.0 - pt) ** gamma * ce
    return focal.mean()


# ─── STRING_GNN Backbone ──────────────────────────────────────────────────────
def load_string_gnn_frozen_prefix():
    """
    Load STRING_GNN, run a single forward pass through mps.0–6 (first 7 layers)
    on the fixed PPI graph, and return the frozen intermediate activations.
    This is more memory-efficient than storing the full model — we only keep the
    intermediate state after mps.6 as a fixed buffer, then fine-tune mps.7+post_mp.

    Returns:
        intermediate_emb : FloatTensor [18870, 256] — output after mps.6
        graph_data       : dict with edge_index, edge_weight (for mps.7 forward)
        node_names       : list of Ensembl IDs aligned with row indices
        gnn_model        : StringGNNModel with mps.7 + post_mp unfrozen
    """
    model_dir = Path(STRING_GNN_DIR)
    gnn_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    graph_data = torch.load(str(model_dir / "graph_data.pt"), map_location="cpu")
    node_names = json.loads((model_dir / "node_names.json").read_text())

    edge_index = graph_data["edge_index"]
    edge_weight = graph_data.get("edge_weight", None)

    # Compute frozen intermediate embedding (after mps.6)
    gnn_model.eval()
    with torch.no_grad():
        # Run forward up to mps.6 (inclusive)
        h = gnn_model.emb.weight  # [18870, 256]
        for i in range(7):  # mps.0 through mps.6
            h = gnn_model.mps[i](h, edge_index, edge_weight)

    intermediate_emb = h.detach().clone().float().cpu()  # [18870, 256], frozen

    # Keep only mps.7 and post_mp trainable; freeze everything else
    for param in gnn_model.parameters():
        param.requires_grad = False
    for param in gnn_model.mps[7].parameters():
        param.requires_grad = True
    for param in gnn_model.post_mp.parameters():
        param.requires_grad = True

    return intermediate_emb, graph_data, node_names, gnn_model


# ─── Dataset & DataModule ─────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    def __init__(self, df, gnn_idx_map, has_labels=True):
        self.pert_ids = df["pert_id"].tolist()
        self.symbols = df["symbol"].tolist()
        # STRING_GNN node index (by Ensembl gene ID)
        self.gnn_indices = [gnn_idx_map.get(pid, -1) for pid in self.pert_ids]
        self.has_labels = has_labels
        if has_labels and "label" in df.columns:
            rows = []
            for lbl_str in df["label"]:
                rows.append([x + 1 for x in json.loads(lbl_str)])  # {-1,0,1} → {0,1,2}
            self.labels = torch.tensor(rows, dtype=torch.long)
        else:
            self.has_labels = False

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "gnn_idx": self.gnn_indices[idx],
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    gnn_indices = torch.tensor([item["gnn_idx"] for item in batch], dtype=torch.long)
    result = {
        "gnn_idx": gnn_indices,
        "pert_id": [item["pert_id"] for item in batch],
        "symbol": [item["symbol"] for item in batch],
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([item["label"] for item in batch])
    return result


class PerturbationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data", micro_batch_size=4, num_workers=4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Build STRING_GNN node index map
        node_names = json.loads((Path(STRING_GNN_DIR) / "node_names.json").read_text())
        gnn_idx_map = {name: i for i, name in enumerate(node_names)}

        dfs = {split: pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")
               for split in ("train", "val", "test")}

        self.train_ds = PerturbationDataset(dfs["train"], gnn_idx_map, True)
        self.val_ds = PerturbationDataset(dfs["val"], gnn_idx_map, True)
        self.test_ds = PerturbationDataset(dfs["test"], gnn_idx_map, True)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.micro_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            collate_fn=collate_fn, persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
            collate_fn=collate_fn, persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
            collate_fn=collate_fn, persistent_workers=self.num_workers > 0,
        )


# ─── Model Components ─────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """Residual block: Linear(d→d*expand)→GELU→Linear(d*expand→d) + skip."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.30):
        super().__init__()
        hidden = dim * expand
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc2.weight)  # Initialize output to 0 for stable start

    def forward(self, x):
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(self.fc2(h))
        return x + h


class GNNBilinearHead(nn.Module):
    """
    6-layer deep residual MLP + bilinear output head.
    Input: [B, 256] STRING_GNN embedding
    Output: [B, 3, N_GENES_OUT] logits
    """

    def __init__(
        self,
        in_dim: int = GNN_DIM,
        hidden_dim: int = 512,
        expand: int = 4,
        n_blocks: int = 6,
        rank: int = 512,
        n_classes: int = N_CLASSES,
        n_genes_out: int = N_GENES_OUT,
        dropout: float = 0.30,
        oov_dim: int = GNN_DIM,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.in_proj.weight)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand, dropout)
            for _ in range(n_blocks)
        ])

        # Bilinear output: [B, hidden] → [B, n_classes, rank] ⊙ [n_genes, rank]
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * rank, bias=True)
        self.out_gene_emb = nn.Embedding(n_genes_out, rank)
        nn.init.normal_(self.out_gene_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)

        # Learnable OOV embedding for genes absent from STRING_GNN
        self.oov_emb = nn.Parameter(torch.zeros(oov_dim))
        nn.init.normal_(self.oov_emb, std=0.01)

        self.n_classes = n_classes
        self.rank = rank
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: [B, in_dim]
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)

        # Bilinear interaction
        proj = self.proj_bilinear(h).view(h.shape[0], self.n_classes, self.rank)
        logits = torch.einsum("bcr,gr->bcg", proj, self.out_gene_emb.weight)
        return logits  # [B, 3, 6640]


class StringGNNPartialFTModel(nn.Module):
    """
    STRING_GNN with partial fine-tuning (mps.7 + post_mp) + deep residual bilinear MLP.

    Architecture:
      - frozen buffer: intermediate activations after mps.6 [18870, 256]
      - trainable: mps.7 (GNNLayer) + post_mp (Linear)
      - head: GNNBilinearHead (6-layer ResidualBlock + bilinear rank-512)
    """

    def __init__(
        self,
        intermediate_emb: torch.Tensor,  # [18870, 256] frozen intermediate state
        graph_data: dict,
        gnn_model,
        hidden_dim: int = 512,
        n_blocks: int = 6,
        rank: int = 512,
        dropout: float = 0.30,
    ):
        super().__init__()

        # Frozen intermediate embedding buffer (output after mps.0–6)
        self.register_buffer("intermediate_emb", intermediate_emb)

        # Partially fine-tuned backbone (mps.7 + post_mp only)
        self.gnn_tail = nn.ModuleDict({
            "mps7": gnn_model.mps[7],
            "post_mp": gnn_model.post_mp,
        })

        # Store graph data as buffers for mps.7 forward
        self.register_buffer("edge_index", graph_data["edge_index"])
        edge_weight = graph_data.get("edge_weight", None)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight.float())
        else:
            self.edge_weight = None

        # Head
        self.head = GNNBilinearHead(
            in_dim=GNN_DIM,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            rank=rank,
            dropout=dropout,
        )

    def get_node_embeddings(self):
        """Run mps.7 + post_mp on the frozen intermediate state."""
        h = self.intermediate_emb  # [18870, 256]
        # Run mps.7
        h = self.gnn_tail["mps7"](h, self.edge_index, self.edge_weight)
        # Apply post_mp
        h = self.gnn_tail["post_mp"](h)
        return h  # [18870, 256]

    def forward(self, gnn_idx):
        # gnn_idx: [B] — STRING_GNN node indices for perturbed genes
        B = gnn_idx.shape[0]

        # Get final node embeddings (fine-tuned mps.7 + post_mp applied)
        node_embs = self.get_node_embeddings()  # [18870, 256]

        # Look up perturbed gene embeddings, with OOV fallback
        valid = gnn_idx >= 0
        safe_idx = gnn_idx.clamp(min=0)
        emb = node_embs[safe_idx]  # [B, 256]

        # Use torch.where for safe OOV masking
        oov_expanded = self.head.oov_emb.float().unsqueeze(0).expand_as(emb)
        oov_mask = (~valid).unsqueeze(-1).expand_as(emb)
        emb = torch.where(oov_mask, oov_expanded, emb.float())

        # Decode through head
        logits = self.head(emb)  # [B, 3, 6640]
        return logits


# ─── DDP gather helper ────────────────────────────────────────────────────────
def _gather_tensors(local_preds, local_labels, device, world_size):
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
    g_preds = [torch.zeros_like(p) for _ in range(world_size)]
    g_labels = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(g_preds, p)
    dist.all_gather(g_labels, l)
    real_preds = torch.cat(
        [g_preds[i][:all_sizes[i].item()].cpu() for i in range(world_size)], 0
    )
    real_labels = torch.cat(
        [g_labels[i][:all_sizes[i].item()].cpu() for i in range(world_size)], 0
    )
    return real_preds, real_labels


# ─── LightningModule ──────────────────────────────────────────────────────────
class PerturbationLitModule(pl.LightningModule):
    def __init__(
        self,
        hidden_dim: int = 512,
        n_blocks: int = 6,
        rank: int = 512,
        dropout: float = 0.30,
        muon_lr: float = 0.005,
        head_lr: float = 5e-4,
        backbone_lr: float = 1e-5,
        weight_decay: float = 2e-3,
        focal_gamma: float = 2.0,
        sgdr_t0: int = 600,
        sgdr_eta_min: float = 1e-6,
        max_steps_total: int = 10000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds = []
        self._val_labels = []
        self._test_preds = []
        self._test_pert_ids = []
        self._test_symbols = []

    def setup(self, stage=None):
        hp = self.hparams

        self.print("Loading STRING_GNN with partial fine-tuning setup...")
        intermediate_emb, graph_data, node_names, gnn_model = load_string_gnn_frozen_prefix()
        self.print(f"STRING_GNN intermediate embeddings shape: {intermediate_emb.shape}")
        self.print(f"Trainable backbone params: mps.7 + post_mp")

        self.model = StringGNNPartialFTModel(
            intermediate_emb=intermediate_emb,
            graph_data=graph_data,
            gnn_model=gnn_model,
            hidden_dim=hp.hidden_dim,
            n_blocks=hp.n_blocks,
            rank=hp.rank,
            dropout=hp.dropout,
        )

        # Cast trainable parameters to float32 for stable optimization
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        self.register_buffer("class_weights", CLASS_WEIGHTS)

    def forward(self, gnn_idx):
        return self.model(gnn_idx)

    def _compute_loss(self, logits, labels):
        return focal_cross_entropy(
            logits, labels,
            class_weights=self.class_weights,
            gamma=self.hparams.focal_gamma,
            label_smoothing=0.0,  # No label smoothing (proven better in tree best nodes)
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["gnn_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["gnn_idx"])
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
        logits = self(batch["gnn_idx"])
        probs = torch.softmax(logits, dim=1)  # [B, 3, 6640]
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            if not hasattr(self, "_test_labels"):
                self._test_labels = []
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs = torch.cat(self._test_preds, dim=0)
        dummy_labels = torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        if hasattr(self, "_test_labels") and self._test_labels:
            dummy_labels = torch.cat(self._test_labels, dim=0)
            del self._test_labels

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
            all_probs = local_probs
            all_labels = dummy_labels
            all_pert = self._test_pert_ids
            all_syms = self._test_symbols

        if self.trainer.is_global_zero:
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"
            seen_ids = set()
            dedup_probs, dedup_labels = [], []
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for i, (pid, sym, probs) in enumerate(
                    zip(all_pert, all_syms, all_probs.numpy())
                ):
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        fh.write(f"{pid}\t{sym}\t{json.dumps(probs.tolist())}\n")
                        dedup_probs.append(probs)
                        dedup_labels.append(all_labels[i].numpy())
            self.print(
                f"Saved test predictions → {pred_path} ({len(seen_ids)} unique samples)"
            )
            if dedup_probs and any(l.any() for l in dedup_labels):
                f1 = compute_per_gene_f1(np.stack(dedup_probs), np.stack(dedup_labels))
                self.print(f"Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Separate parameters into 3 groups for MuonWithAuxAdam
        # Group 1 (Muon): 2D hidden weight matrices in ResidualBlock (fc1, fc2)
        muon_params = []
        # Group 2 (AdamW head): embeddings, projection biases, norms, out_gene_emb, oov_emb
        adamw_head_params = []
        # Group 3 (AdamW backbone): mps.7 and post_mp trainable params
        adamw_backbone_params = []

        head_param_ids = set()
        backbone_param_ids = set()

        # Identify backbone params (mps.7 + post_mp)
        for n, p in self.model.gnn_tail.named_parameters():
            if p.requires_grad:
                adamw_backbone_params.append(p)
                backbone_param_ids.add(id(p))

        # Classify head params
        for n, p in self.model.head.named_parameters():
            if not p.requires_grad:
                continue
            # Muon targets: fc1.weight, fc2.weight (2D matrices in hidden layers)
            if p.ndim >= 2 and any(x in n for x in ["fc1.weight", "fc2.weight"]):
                muon_params.append(p)
            else:
                adamw_head_params.append(p)
            head_param_ids.add(id(p))

        # Also add oov_emb (1D) to adamw_head
        for n, p in self.model.named_parameters():
            if p.requires_grad and id(p) not in head_param_ids and id(p) not in backbone_param_ids:
                adamw_head_params.append(p)

        self.print(
            f"Optimizer groups: Muon={len(muon_params)} params, "
            f"AdamW-head={len(adamw_head_params)} params, "
            f"AdamW-backbone={len(adamw_backbone_params)} params"
        )

        try:
            from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
            param_groups = [
                dict(params=muon_params, use_muon=True, lr=hp.muon_lr,
                     weight_decay=0.0, momentum=0.95),
                dict(params=adamw_head_params, use_muon=False, lr=hp.head_lr,
                     betas=(0.9, 0.95), weight_decay=hp.weight_decay),
                dict(params=adamw_backbone_params, use_muon=False, lr=hp.backbone_lr,
                     betas=(0.9, 0.95), weight_decay=0.0),
            ]
            # Use MuonWithAuxAdam for distributed (multi-GPU) and
            # SingleDeviceMuonWithAuxAdam for single-GPU (avoids dist.get_world_size() call)
            n_gpus_runtime = int(os.environ.get("WORLD_SIZE", 1))
            if n_gpus_runtime > 1 and dist.is_available() and dist.is_initialized():
                optimizer = MuonWithAuxAdam(param_groups)
                self.print("Using MuonWithAuxAdam (distributed)")
            else:
                optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
                self.print("Using SingleDeviceMuonWithAuxAdam (single-device)")
        except ImportError:
            self.print("MuonWithAuxAdam not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(
                [
                    {"params": muon_params, "lr": hp.head_lr},
                    {"params": adamw_head_params, "lr": hp.head_lr},
                    {"params": adamw_backbone_params, "lr": hp.backbone_lr},
                ],
                weight_decay=hp.weight_decay,
            )

        # SGDR: CosineAnnealingWarmRestarts with T_0 steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hp.sgdr_t0,
            T_mult=1,
            eta_min=hp.sgdr_eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save trainable params + small buffers; exclude large intermediate_emb (~18.5 MB)."""
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        # Include class_weights buffer; exclude large intermediate_emb
        buffer_keys = {
            prefix + n for n, _ in self.named_buffers()
            if "intermediate_emb" not in n and "edge_index" not in n and "edge_weight" not in n
        }
        trainable_sd = {
            k: v for k, v in full_sd.items()
            if k in trainable_keys or k in buffer_keys
        }
        total = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving ckpt: {trained}/{total} trainable params ({100 * trained / total:.1f}%)"
        )
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        """Load trainable params; intermediate_emb + graph buffers recomputed in setup()."""
        return super().load_state_dict(state_dict, strict=False)


# ─── Argument Parsing ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="node1-1-3-1-1-1: STRING_GNN Partial FT + Deep Bilinear MLP + Muon + SGDR"
    )
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=6)
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--muon-lr", type=float, default=0.005)
    p.add_argument("--head-lr", type=float, default=5e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=2e-3)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--sgdr-t0", type=int, default=600)
    p.add_argument("--sgdr-eta-min", type=float, default=1e-6)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=128)
    p.add_argument("--max-epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true", default=False)
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────
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

    # Compute total steps for LR schedule reference
    _train_df_size = pd.read_csv(
        Path(args.data_dir) / "train.tsv", sep="\t", usecols=["pert_id"]
    ).shape[0]
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    steps_per_epoch = _train_df_size // (args.micro_batch_size * n_gpus)
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)
    max_steps_total = effective_steps_per_epoch * args.max_epochs

    lit = PerturbationLitModule(
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        rank=args.rank,
        dropout=args.dropout,
        muon_lr=args.muon_lr,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        sgdr_t0=args.sgdr_t0,
        sgdr_eta_min=args.sgdr_eta_min,
        max_steps_total=max(max_steps_total, 1),
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    es_cb = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-4)
    lr_cb = LearningRateMonitor(logging_interval="step")
    pb_cb = TQDMProgressBar(refresh_rate=10)
    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps = -1
    limit_train_batches = 1.0
    limit_val_batches = 1.0
    limit_test_batches = 1.0
    fast_dev_run = False

    if args.debug_max_step is not None:
        max_steps = args.debug_max_step
        limit_train_batches = args.debug_max_step
        limit_val_batches = 2
        limit_test_batches = 2
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
        gradient_clip_val=1.0,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
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
    test_results = trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(f"Test results from trainer: {test_results}\n")


if __name__ == "__main__":
    main()
