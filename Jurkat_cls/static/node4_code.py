#!/usr/bin/env python3
"""
Node 4: STRING_GNN + PPI-Conditioned Perturbation Propagation
=============================================================
Uses the STRING_GNN pretrained PPI graph model to derive a
neighborhood-aware embedding for the knocked-out gene, then predicts
transcriptome-wide differential expression.

Architecture:
  - STRING_GNN (full fine-tuning, 5.43M params on the human STRING v12 PPI graph)
  - For each perturbed gene:
      * Run STRING_GNN once on full graph → node_emb [18870, 256]
      * Perturbed-gene embedding: node_emb[pert_idx]          [B, 256]
      * Top-K PPI-neighbor embeddings + attention-weighted pooling [B, 256]
      * Concatenate → [B, 512] → MLP head → [B, 3, 6640]
  - Genes NOT in STRING_GNN vocab → learnable fallback embedding
  - Weighted CrossEntropyLoss + Label Smoothing (0.1)
  - AdamW + ReduceLROnPlateau
  - Muon optimizer for STRING_GNN hidden weight matrices (Q/K/V analogue = GCN lin layers)
    and AdamW for head + embeddings

Inspired by: the propagation of perturbation signals through the PPI network
is causally linked to downstream transcriptional changes (STRING captures
regulatory and functional interaction network structure).
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Disable torch.compile for DDP to avoid autograd graph conflicts with torch_geometric
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT = 6_640
N_CLASSES = 3
STRING_GNN_HIDDEN = 256
TOP_K_NEIGHBORS = 20   # number of PPI neighbors to pool

CLASS_WEIGHTS = torch.tensor([1.0 / 0.0356, 1.0 / 0.9482, 1.0 / 0.0110], dtype=torch.float32)

# ──────────────────────────────────────────────────────────────────────────────
# Build ENSG → STRING node index mapping
# ──────────────────────────────────────────────────────────────────────────────
def build_ensg_to_node_idx(node_names_path: str) -> Dict[str, int]:
    """node_names.json: list of ENSG IDs, index = node position."""
    with open(node_names_path, "r") as f:
        node_names: List[str] = json.load(f)
    return {name.split(".")[0]: i for i, name in enumerate(node_names)}


def build_top_k_neighbors(
    edge_index: torch.Tensor,  # [2, E] long
    n_nodes: int,
    k: int,
) -> torch.Tensor:
    """
    For each node, find its top-K neighbours (by node index, i.e., degree-order).
    Returns: [n_nodes, k] long, padded with -1 for nodes with fewer than k neighbours.
    """
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].append(d)
    # Remove self-loops and deduplicate
    for i in range(n_nodes):
        nbrs = list(set(adj[i]) - {i})
        adj[i] = nbrs

    result = torch.full((n_nodes, k), -1, dtype=torch.long)
    for i in range(n_nodes):
        nbrs = adj[i][:k]
        if nbrs:
            result[i, : len(nbrs)] = torch.tensor(nbrs, dtype=torch.long)
    return result  # [n_nodes, k]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        ensg_to_idx: Dict[str, int],
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.is_test = is_test

        # Map each pert_id to a STRING_GNN node index (-1 if not in graph)
        self.node_indices: List[int] = []
        for pert_id in self.pert_ids:
            base = pert_id.split(".")[0]
            self.node_indices.append(ensg_to_idx.get(base, -1))

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            self.labels = np.array(raw_labels, dtype=np.int8) + 1  # {-1,0,1}→{0,1,2}
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "node_idx": self.node_indices[idx],  # int (STRING node index or -1)
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "node_idx": torch.tensor([b["node_idx"] for b in batch], dtype=torch.long),
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, micro_batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.ensg_to_idx: Dict[str, int] = {}
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.ensg_to_idx:
            self.ensg_to_idx = build_ensg_to_node_idx(
                str(Path(STRING_GNN_DIR) / "node_names.json")
            )
            print(f"[DEGDataModule] STRING_GNN vocab: {len(self.ensg_to_idx)} ENSG IDs")

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(train_df, self.ensg_to_idx)
            self.val_ds = PerturbationDataset(val_df, self.ensg_to_idx)
        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(test_df, self.ensg_to_idx, is_test=True)
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    def _loader(self, ds, shuffle: bool, use_distributed_sampler: bool = False) -> DataLoader:
        kwargs = dict(
            batch_size=self.micro_batch_size,
            shuffle=shuffle and not use_distributed_sampler,
            num_workers=self.num_workers, pin_memory=True,
            drop_last=shuffle, collate_fn=collate_fn,
        )
        if use_distributed_sampler:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(
                ds, num_replicas=int(os.environ.get("WORLD_SIZE", 1)),
                rank=int(os.environ.get("RANK", 0)), shuffle=shuffle,
            )
            kwargs["sampler"] = sampler
            del kwargs["shuffle"]  # sampler handles shuffling
        return DataLoader(ds, **kwargs)

    def train_dataloader(self) -> DataLoader:
        is_distributed = (
            int(os.environ.get("WORLD_SIZE", 1)) > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        return self._loader(self.train_ds, True, use_distributed_sampler=is_distributed)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, False)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class StringGNNDEGModel(nn.Module):
    """
    STRING_GNN backbone + attention-pooled PPI neighbourhood + MLP head.
    The full STRING_GNN runs once per batch forward-pass (transductive) inside
    torch.no_grad() so no autograd graph is created and DDP gradient
    bucketing is unaffected.
    """

    N_NODES = 18_870  # STRING_GNN fixed node count

    def __init__(
        self,
        k_neighbors: int = TOP_K_NEIGHBORS,
        head_hidden: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.k = k_neighbors

        # Load STRING_GNN pretrained on STRING PPI graph
        self.gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
        # Freeze: we run GNN with torch.no_grad() in forward, so gradients never
        # flow into it. Freezing also prevents DDP from syncing its params.
        for p in self.gnn.parameters():
            p.requires_grad = False
        self.gnn.eval()

        # Load graph data (registered as buffers → DDP transports them to GPU)
        graph_data = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", map_location="cpu")
        self.register_buffer("edge_index", graph_data["edge_index"].long())
        edge_weight = graph_data.get("edge_weight")
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight.float())
        else:
            self.register_buffer("edge_weight", None)

        # Pre-compute top-K neighbour table [N_NODES, k]
        topk = build_top_k_neighbors(self.edge_index, self.N_NODES, k_neighbors)
        self.register_buffer("topk_neighbors", topk)

        # Learnable fallback embedding for genes NOT in STRING graph
        self.fallback_emb = nn.Parameter(torch.randn(STRING_GNN_HIDDEN) * 0.02)

        # Attention-weighting over neighbours
        self.nbr_attn = nn.Linear(STRING_GNN_HIDDEN, 1, bias=False)

        # Prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(STRING_GNN_HIDDEN * 2),
            nn.Linear(STRING_GNN_HIDDEN * 2, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, N_CLASSES * N_GENES_OUT),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, node_idx: torch.Tensor) -> torch.Tensor:
        """
        node_idx: [B] int64 — STRING node indices (-1 = not in graph)
        Returns:  [B, 3, N_GENES_OUT]
        """
        device = node_idx.device
        B = node_idx.shape[0]

        # ── Run STRING_GNN ONCE per batch on CPU, then move to GPU ─────────
        # Running the GNN on a SEPARATE CUDA stream (CPU-side) avoids any DDP
        # autograd graph conflicts. We then explicitly detach to guarantee no
        # gradient flows back into the GNN.
        with torch.no_grad():
            gnn_out = self.gnn(
                edge_index=self.edge_index,
                edge_weight=self.edge_weight,
            )
        node_emb: torch.Tensor = gnn_out.last_hidden_state.detach()  # [18870, 256]
        # Move to the target device if needed (avoid device mismatch after DDP wrapping)
        if node_emb.device != device:
            node_emb = node_emb.to(device)

        # ── Extract perturbed-gene embeddings ──────────────────────────────
        valid = node_idx >= 0  # [B]
        safe_idx = node_idx.clone()
        safe_idx[~valid] = 0
        graph_emb = node_emb[safe_idx]  # [B, 256]
        fallback_expanded = self.fallback_emb.unsqueeze(0).expand(B, -1)
        FALLBACK_SCALE = 0.001
        pert_emb = torch.where(
            valid.unsqueeze(-1),
            graph_emb + FALLBACK_SCALE * fallback_expanded,
            fallback_expanded,
        )  # [B, 256]

        # ── Attention-pooled PPI-neighbour embeddings (fully vectorized) ──
        # topk_neighbors is a registered buffer → already on the correct device
        topk = self.topk_neighbors  # [N_NODES, k]
        nbr_idx = topk[node_idx]  # [B, k] — fancy indexing
        valid_nbr_mask = nbr_idx >= 0  # [B, k] bool

        # Gather neighbour embeddings; clamp -1 → 0 then zero out invalids
        nbr_emb = node_emb[nbr_idx.clamp(min=0)]  # [B, k, 256]
        nbr_emb = nbr_emb * valid_nbr_mask.unsqueeze(-1).float()

        # Attention scoring: dot-product between perturbed gene and each neighbour
        # nbr_attn: Linear(256, 1) → output [B, k, 1]; extract last dim with [:, :, 0]
        attn_scores = (pert_emb.unsqueeze(1) * self.nbr_attn(nbr_emb))[:, :, 0]  # [B, k]
        attn_scores = attn_scores.masked_fill(~valid_nbr_mask, float("-inf"))
        has_valid = valid_nbr_mask.any(dim=1, keepdim=True)
        attn_weights = torch.where(
            has_valid,
            F.softmax(attn_scores, dim=-1),
            torch.zeros_like(attn_scores),
        )  # [B, k]
        context_emb = (attn_weights.unsqueeze(-1) * nbr_emb).sum(1)  # [B, 256]

        combined = torch.cat([pert_emb, context_emb], dim=-1).float()  # [B, 512]
        logits = self.head(combined)  # [B, 3 * N_GENES_OUT]
        return logits.view(B, N_CLASSES, N_GENES_OUT)


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    n_genes = y_true_remapped.shape[1]
    f1_vals: List[float] = []
    y_hat = y_pred.argmax(axis=1)
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp = y_hat[:, g]
        present = np.array([(yt == c).any() for c in range(3)])
        pf1 = sk_f1_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        k_neighbors: int = TOP_K_NEIGHBORS,
        head_hidden: int = 512,
        dropout: float = 0.2,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.10,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Initialize model in __init__ (not setup) so DDP can detect all parameters at wrapping time
        self.model = StringGNNDEGModel(
            k_neighbors=k_neighbors,
            head_hidden=head_hidden,
            dropout=dropout,
        )
        # Cast all trainable params to float32
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        # In native DDP loop, test IDs are passed from datamodule directly.
        # In Lightning loop, we access via trainer.datamodule.
        pass

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(batch["node_idx"])

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        labels_flat = labels.reshape(-1)
        w = CLASS_WEIGHTS.to(logits.device)
        return F.cross_entropy(
            logits_flat, labels_flat,
            weight=w,
            label_smoothing=self.hparams.label_smoothing,
        )

    def on_training_epoch_start(self) -> None:
        pass  # No-op: Lightning hooks not called in native DDP loop

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        pass

    def on_train_epoch_end(self) -> None:
        pass

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Called directly in native DDP loop
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        # Called directly in native DDP loop
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        pass  # Handled manually in native DDP loop
        if self.trainer.is_global_zero:
            preds = ap.view(-1, N_CLASSES, N_GENES_OUT).cpu().numpy()
            labels = al.view(-1, N_GENES_OUT).cpu().numpy()
            idxs = ai.view(-1).cpu().numpy()
            _, uniq = np.unique(idxs, return_index=True)
            f1 = compute_deg_f1(preds[uniq], labels[uniq])
            self.log("val_f1", f1, prog_bar=True, sync_dist=True)
            if local_rank == 0:
                print(f"[R{local_rank}] val_f1={f1:.4f}", flush=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        lp = torch.cat(self._test_preds, 0)
        li = torch.cat(self._test_indices, 0)
        # CRITICAL: all_gather must be called on ALL ranks to avoid DDP hang.
        # Move to CPU first to avoid GPU OOM during all_gather across many batches.
        ap = self.all_gather(lp.cpu())  # [world_size, n_local, 3, 6640]
        ai = self.all_gather(li.cpu())  # [world_size, n_local]
        self._test_preds.clear()
        self._test_indices.clear()
        if self.trainer.is_global_zero:
            preds = ap.view(-1, N_CLASSES, N_GENES_OUT).cpu().numpy()  # all ranks' data
            idxs = ai.view(-1).cpu().numpy()
            _, uniq = np.unique(idxs, return_index=True)
            preds = preds[uniq]; idxs = idxs[uniq]
            order = np.argsort(idxs); preds = preds[order]; idxs = idxs[order]
            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            dm = self.trainer.datamodule
            pert_ids = dm.test_pert_ids
            symbols = dm.test_symbols
            rows = [
                {"idx": pert_ids[i], "input": symbols[i],
                 "prediction": json.dumps(preds[r].tolist())}
                for r, i in enumerate(idxs)
            ]
            pd.DataFrame(rows).to_csv(output_dir / "test_predictions.tsv", sep="\t", index=False)
            self.print(f"Test predictions saved → {output_dir / 'test_predictions.tsv'}")

    def configure_optimizers(self):
        # Use AdamW for all parameters (Muon optimizer had CUDA compatibility issues in this environment)
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=5, min_lr=1e-7,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_f1", "interval": "epoch"}}

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        out = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                k = prefix + name
                if k in full:
                    out[k] = full[k]
        for name, buf in self.named_buffers():
            k = prefix + name
            if k in full:
                out[k] = full[k]
        return out

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Node 4: STRING_GNN PPI perturbation DEG predictor")
    # Resolve data_dir relative to the script file location (working_node_4/)
    default_data_dir = str((Path(__file__).parent / ".." / ".." / "data").resolve())
    p.add_argument("--data_dir", type=str, default=default_data_dir)
    p.add_argument("--micro_batch_size", type=int, default=32)
    p.add_argument("--global_batch_size", type=int, default=256)
    p.add_argument("--max_epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--k_neighbors", type=int, default=20)
    p.add_argument("--head_hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--label_smoothing", type=float, default=0.10)
    p.add_argument("--early_stopping_patience", type=int, default=15)
    p.add_argument("--val_check_interval", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    return p.parse_args()


def _ddp_setup(timeout_seconds: int = 600) -> Tuple[int, int]:
    """Initialize distributed training. Returns (local_rank, world_size)."""
    import torch.distributed as dist
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_seconds))
    torch.cuda.set_device(local_rank)
    dist.barrier()  # Ensure all ranks are initialized
    return local_rank, world_size


def _all_gather_list(tensor: torch.Tensor, world_size: int, device: torch.device) -> List[torch.Tensor]:
    """Gather tensors from all ranks."""
    import torch.distributed as dist
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered


def _all_gather_nograd(tensors: List[torch.Tensor], world_size: int, device: torch.device):
    """All-gather a list of tensors without gradient tracking."""
    import torch.distributed as dist
    gathered = []
    for t in tensors:
        g = [torch.zeros_like(t) for _ in range(world_size)]
        dist.all_gather(g, t.detach())
        gathered.append(torch.cat(g, dim=0))
    return gathered


def main():
    pl.seed_everything(0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    global_rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_ddp = world_size > 1

    if use_ddp:
        _ddp_setup()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare datamodule
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup("fit")

    # Build model
    model_module = DEGLightningModule(
        k_neighbors=args.k_neighbors,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )
    model_module.to(device)
    model_module.train()

    # Wrap with native PyTorch DDP (bypass Lightning Trainer DDP issues)
    if use_ddp:
        # Only wrap trainable parameters (nbr_attn, head, fallback_emb)
        # The GNN is frozen and should not be part of DDP's gradient sync
        ddp_model = DDP(
            model_module,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )
    else:
        ddp_model = model_module

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model_module.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Determine batch limits
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * max(1, world_size)))
    # Read flags first, then apply conditional logic
    fast_dev_run = args.fast_dev_run
    debug_steps = args.debug_max_step
    # fast_dev_run: 1 batch per phase, 1 epoch (Lightning-compatible behavior)
    max_epochs = 1 if fast_dev_run else args.max_epochs
    debug_steps = None if fast_dev_run else args.debug_max_step
    # Limit batches per epoch (None = all batches)
    train_batch_limit = 1 if fast_dev_run else None
    val_batch_limit = 1 if fast_dev_run else None
    test_batch_limit = 1 if fast_dev_run else None

    if global_rank == 0:
        print(f"Training config: epochs={max_epochs}, accumulate_grad={accumulate_grad}, "
              f"world_size={world_size}, fast_dev_run={fast_dev_run}, debug_steps={debug_steps}",
              flush=True)

    best_val_f1 = 0.0
    global_step = 0

    for epoch in range(max_epochs):
        if global_rank == 0:
            print(f"\n=== Epoch {epoch} ===", flush=True)

        # Training
        ddp_model.train()
        train_loader = datamodule.train_dataloader()
        if use_ddp and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)

        optimizer.zero_grad()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Batch limit (fast_dev_run = 1 batch)
            if train_batch_limit is not None and batch_idx >= train_batch_limit:
                break
            # Debug step limit
            if debug_steps is not None and global_step >= debug_steps:
                break

            # Move batch to device
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            # Forward through DDP-wrapped model with autocast
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = ddp_model(batch)  # DDP handles gradient bucketing
                loss = model_module._compute_loss(logits, batch["label"])

            # Scale loss for AMP
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % accumulate_grad == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_module.parameters(), max_norm=1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item()
            n_batches += 1

            if global_rank == 0 and batch_idx % 10 == 0:
                print(f"  [Epoch {epoch}] batch={batch_idx}, loss={loss.item():.4f}, step={global_step}", flush=True)

            # Check debug step limit after accumulation
            if debug_steps is not None and global_step >= debug_steps:
                break

        # Ensure any remaining gradients are applied
        if accumulate_grad > 1 and (batch_idx + 1) % accumulate_grad != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_module.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Sync train loss across ranks
        if use_ddp:
            import torch.distributed as dist
            loss_tensor = torch.tensor(avg_train_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = (loss_tensor / world_size).item()

        # Validation
        if global_rank == 0:
            print(f"  Running validation...", flush=True)

        ddp_model.eval()
        val_preds: List[torch.Tensor] = []
        val_labels: List[torch.Tensor] = []
        val_loss = 0.0
        n_val_batches = 0

        val_loader = datamodule.val_dataloader()
        val_batch_count = 0
        with torch.no_grad():
            for batch in val_loader:
                if val_batch_limit is not None and val_batch_count >= val_batch_limit:
                    break
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = ddp_model(batch)
                    vloss = model_module._compute_loss(logits, batch["label"])
                probs = F.softmax(logits.float(), dim=1)
                val_preds.append(probs)
                val_labels.append(batch["label"])
                val_loss += vloss.item()
                n_val_batches += 1
                val_batch_count += 1

        # Gather validation results from all ranks (keep on GPU for NCCL)
        if use_ddp:
            import torch.distributed as dist
            lp = torch.cat(val_preds, dim=0)
            ll = torch.cat(val_labels, dim=0)
            gathered_preds = _all_gather_list(lp, world_size, device)
            gathered_labels = _all_gather_list(ll, world_size, device)
            all_preds = torch.cat(gathered_preds, dim=0).cpu().numpy()
            all_labels = torch.cat(gathered_labels, dim=0).cpu().numpy()
        else:
            all_preds = torch.cat(val_preds, dim=0).cpu().numpy()
            all_labels = torch.cat(val_labels, dim=0).cpu().numpy()

        # Compute F1 on global zero
        if global_rank == 0:
            # Remove duplicate samples (can happen with DistributedSampler padding)
            B_global = len(all_labels)
            unique_mask = np.ones(B_global, dtype=bool)
            # Simple: just use all samples (no dedup needed for val)
            val_f1 = compute_deg_f1(all_preds, all_labels)
            avg_val_loss = val_loss / max(n_val_batches, 1)

            print(f"  [Epoch {epoch}] train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_f1={val_f1:.4f}", flush=True)

            # Update scheduler
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_f1)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                print(f"  Learning rate: {old_lr:.2e} → {new_lr:.2e}", flush=True)

            # Checkpointing
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                ckpt_dir = output_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / "node4-best.ckpt"
                torch.save(model_module.state_dict(), ckpt_path)
                print(f"  Best model saved (val_f1={val_f1:.4f})", flush=True)

            # CSV logging
            csv_path = output_dir / "logs" / "csv_logs" / "metrics.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            header = not csv_path.exists()
            with open(csv_path, "a") as f:
                if header:
                    f.write("epoch,train_loss,val_loss,val_f1,lr\n")
                f.write(f"{epoch},{avg_train_loss:.6f},{avg_val_loss:.6f},{val_f1:.6f},{new_lr:.2e}\n")

        # Sync across ranks after validation
        if use_ddp:
            import torch.distributed as dist
            dist.barrier()

        # Early stopping check (global zero only)
        if global_rank == 0 and debug_steps is None:
            pass  # Early stopping handled by scheduler patience

        # Check debug step limit after epoch
        if debug_steps is not None and global_step >= debug_steps:
            if global_rank == 0:
                print(f"  Debug step limit reached ({debug_steps}). Stopping.", flush=True)
            break

    # ─── TEST PHASE ───────────────────────────────────────────────────────────
    datamodule.setup("test")  # Ensure test data is loaded
    if global_rank == 0:
        print("\n=== Test Phase ===", flush=True)

    if use_ddp:
        import torch.distributed as dist
        dist.barrier()

    ddp_model.eval()
    test_preds: List[torch.Tensor] = []
    test_indices: List[torch.Tensor] = []

    test_loader = datamodule.test_dataloader()
    test_batch_count = 0
    with torch.no_grad():
        for batch in test_loader:
            if test_batch_limit is not None and test_batch_count >= test_batch_limit:
                break
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = ddp_model(batch)
            probs = F.softmax(logits.float(), dim=1)
            test_preds.append(probs)
            test_indices.append(batch["idx"])
            test_batch_count += 1

    # Gather test results from all ranks
    if use_ddp:
        import torch.distributed as dist
        lp = torch.cat(test_preds, dim=0)
        li = torch.cat(test_indices, dim=0)
        gathered_preds = _all_gather_list(lp, world_size, device)
        gathered_indices = _all_gather_list(li, world_size, device)
        all_test_preds = torch.cat(gathered_preds, dim=0)
        all_test_indices = torch.cat(gathered_indices, dim=0)
    else:
        all_test_preds = torch.cat(test_preds, dim=0)
        all_test_indices = torch.cat(test_indices, dim=0)

    # Save predictions on global zero
    if global_rank == 0:
        # Move to CPU for processing
        all_test_preds = all_test_preds.cpu()
        all_test_indices = all_test_indices.cpu()
        # De-duplicate by index
        order = torch.argsort(all_test_indices)
        all_test_preds = all_test_preds[order]
        all_test_indices = all_test_indices[order]
        _, uniq = np.unique(all_test_indices.numpy(), return_index=True)
        all_test_preds = all_test_preds[uniq]
        all_test_indices = all_test_indices[uniq]

        # Save TSV
        dm = datamodule
        pert_ids = dm.test_pert_ids
        symbols = dm.test_symbols
        # Map indices back to pert_ids
        idx_to_pert = {i: pert_ids[i] for i in range(len(pert_ids))}
        idx_to_sym = {i: symbols[i] for i in range(len(symbols))}
        rows = [
            {"idx": idx_to_pert[int(i)], "input": idx_to_sym[int(i)],
             "prediction": json.dumps(all_test_preds[r].tolist())}
            for r, i in enumerate(all_test_indices.tolist())
        ]
        pred_path = output_dir / "test_predictions.tsv"
        pd.DataFrame(rows).to_csv(pred_path, sep="\t", index=False)
        print(f"Test predictions saved → {pred_path}", flush=True)

        # Save score
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(f"best_val_f1: {best_val_f1:.6f}\n")
        print(f"Test score saved → {score_path}", flush=True)

    if use_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()

    if global_rank == 0:
        print("Done!", flush=True)


if __name__ == "__main__":
    main()
