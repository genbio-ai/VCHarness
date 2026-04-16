"""
Node 4-1 – STRING_GNN + cond_emb Perturbation Injection + Multi-Layer Fusion + Enhanced Head

Architecture:
  - STRING_GNN loaded from /home/Models/STRING_GNN (full fine-tuning, ~5.43M)
  - cond_emb perturbation injection: for each sample, a cond_emb tensor of shape
    [N_nodes, 256] is created with a 1.0 activation at the perturbed gene node,
    so GNN message passing propagates the perturbation signal through the PPI graph.
    Since all samples in a batch may differ, we run unique perturbations in parallel:
    we construct a stacked cond_emb of unique perturbations, batch-process them by
    expanding edge_index/edge_weight for sparse batching, OR serially run per-unique.
    Implementation: serial per-unique forward pass (typically 1-8 unique per batch).
  - output_hidden_states=True: extract embeddings from all 9 states (1 init + 8 GNN)
  - Learned layer attention weights: softmax over 9 layers → weighted sum → [N_nodes, 256]
  - Enhanced nonlinear interaction head:
      fused_emb [B, 256] → LayerNorm → Linear(256→512) → GELU → Dropout(0.15)
                        → Linear(512→3×256) → reshape → [B, 3, 256]
      output_gene_emb [6640, 256]  (learnable)
      logits = [B, 3, 256] @ [256, 6640] → [B, 3, 6640]
  - Weighted cross-entropy with label smoothing (0.05) + class weights [12.28, 1.12, 33.33]

Key improvements vs Node 4 (parent):
  1. cond_emb injection: PPI graph propagates perturbation signal from node-of-interest,
     directly addressing the root-cause representational capacity ceiling
  2. Multi-layer hidden state fusion: attention-weighted aggregation of all 9 GNN layer
     outputs captures local (early layers) and global (late layers) PPI topology
  3. Nonlinear MLP head: 2-layer MLP before bilinear projection adds nonlinearity
     and expressivity that a single linear projection lacked
  4. Stronger regularization: weight_decay=1e-3 (vs 1e-4) + dropout=0.15
     to mitigate train-val calibration gap observed in Node 4
  5. Label smoothing=0.05 for better probability calibration
  6. CosineAnnealingLR: smooth learning rate decay avoids ReduceLROnPlateau's
     aggressive step-wise decay that caused premature convergence in Node 4
  7. Increased patience=25 to give extra time for the conditioned GNN to find its optimum
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

STRING_GNN_DIR  = "/home/Models/STRING_GNN"
N_GENES_OUT     = 6640
N_CLASSES       = 3
GNN_DIM         = 256
N_GNN_LAYERS    = 8    # 8 message-passing layers
N_HIDDEN_STATES = 9    # 1 (init emb) + 8 GNN layers

# Class weights: inverse frequency of {down=-1, neutral=0, up=+1}
# Train distribution: 8.14% down, 88.86% neutral, 3.00% up
CLASS_WEIGHTS = torch.tensor([12.28, 1.12, 33.33], dtype=torch.float32)


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
        print(f"[Node4-1] {n_unknown}/{total} samples not found in STRING_GNN graph "
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


# ─── Model ────────────────────────────────────────────────────────────────────

class StringGNNCondBilinearModel(nn.Module):
    """
    STRING_GNN backbone with:
      1. cond_emb perturbation injection at the perturbed gene node
      2. Multi-layer hidden state attention pooling (9 layers → weighted sum)
      3. Enhanced 2-layer MLP interaction head before bilinear projection
    """

    def __init__(
        self,
        edge_index:   torch.Tensor,
        edge_weight:  Optional[torch.Tensor],
        n_nodes:      int,
        n_genes_out:  int   = N_GENES_OUT,
        n_classes:    int   = N_CLASSES,
        gnn_dim:      int   = GNN_DIM,
        head_hidden:  int   = 512,
        dropout:      float = 0.15,
    ):
        super().__init__()
        self.n_nodes   = n_nodes
        self.gnn_dim   = gnn_dim

        self.gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)

        # Register graph tensors as buffers (auto device transfer)
        self.register_buffer("edge_index", edge_index)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight = None

        # Fallback for genes absent from STRING_GNN
        self.fallback_emb = nn.Parameter(torch.randn(gnn_dim) * 0.02)

        # Layer attention weights for multi-layer fusion: 9 scalars
        self.layer_attention = nn.Parameter(torch.zeros(N_HIDDEN_STATES))

        # Learnable output-gene embeddings [6640, 256]
        self.out_gene_emb = nn.Embedding(n_genes_out, gnn_dim)
        nn.init.xavier_uniform_(self.out_gene_emb.weight)

        # Enhanced 2-layer MLP head:
        # LayerNorm → Linear(gnn_dim → head_hidden) → GELU → Dropout
        # → Linear(head_hidden → n_classes × gnn_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(gnn_dim),
            nn.Linear(gnn_dim, head_hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes * gnn_dim, bias=False),
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Node4-1] Trainable params: {n_trainable:,}")

    def _fused_emb_for_nodes(
        self,
        unique_node_idxs: torch.Tensor,  # [U] unique node indices (may include -1)
    ) -> torch.Tensor:
        """
        For each unique node index, run the STRING_GNN with a cond_emb signal
        at that node, extract multi-layer attention-fused embedding.
        Returns: [U, gnn_dim]
        """
        device = self.edge_index.device
        ew     = self.edge_weight
        attn_w = F.softmax(self.layer_attention, dim=0)  # [9]

        result = []
        for ui in range(unique_node_idxs.shape[0]):
            gidx = unique_node_idxs[ui].item()

            # Build cond_emb: zeros everywhere, unit at perturbed node (if known)
            # Match dtype of self.gnn.emb.weight to avoid bf16/fp32 mismatch
            emb_dtype = self.gnn.emb.weight.dtype
            cond = torch.zeros(self.n_nodes, self.gnn_dim,
                               device=device, dtype=emb_dtype)
            if gidx >= 0:
                cond[gidx, :] = 1.0

            # Full-graph forward with perturbation conditioning + all hidden states
            out = self.gnn(
                edge_index=self.edge_index,
                edge_weight=ew,
                cond_emb=cond,
                output_hidden_states=True,
            )
            # hidden_states: tuple of 9 tensors [N_nodes, 256]
            hs = torch.stack(out.hidden_states, dim=0)  # [9, N_nodes, 256]

            # Attention-weighted multi-layer fusion → [N_nodes, 256]
            fused = (hs * attn_w.view(-1, 1, 1)).sum(dim=0)  # [N_nodes, 256]

            # Extract embedding at perturbed node
            safe_idx  = max(gidx, 0)
            emb_i     = fused[safe_idx, :]  # [256]
            result.append(emb_i)

        return torch.stack(result, dim=0)  # [U, 256]

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_indices: [B] long – STRING_GNN node indices (-1 for unknown)
        Returns:
            logits: [B, 3, 6640]
        """
        # Compute unique conditioned embeddings (reduces redundant GNN passes)
        unique_idxs, inverse = torch.unique(node_indices, return_inverse=True)
        unique_embs = self._fused_emb_for_nodes(unique_idxs)  # [U, 256]
        pert_emb    = unique_embs[inverse]                     # [B, 256]

        # Replace unknown genes (idx=-1 → fallback at cond=0) with explicit fallback
        # (the gidx<0 branch in _fused_emb_for_nodes already handles this via cond=0,
        #  but we also override the selected embedding position with the learned fallback
        #  to ensure the fallback path is explicitly distinct)
        known_mask = (node_indices >= 0)
        if not known_mask.all():
            fallback = self.fallback_emb.unsqueeze(0).expand_as(pert_emb)
            pert_emb = torch.where(
                known_mask.unsqueeze(-1).expand_as(pert_emb),
                pert_emb, fallback
            )

        pert_emb = pert_emb.float()  # ensure float32

        # Enhanced MLP head: [B, 256] → [B, n_classes × 256]
        pert_proj = self.head(pert_emb).view(-1, N_CLASSES, self.gnn_dim)  # [B, 3, 256]

        # Bilinear: [B, 3, 256] @ [256, 6640] → [B, 3, 6640]
        out_embs  = self.out_gene_emb.weight                  # [6640, 256]
        logits    = torch.matmul(pert_proj, out_embs.T)
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


# ─── LightningModule ──────────────────────────────────────────────────────────

class StringGNNCondLitModule(pl.LightningModule):

    def __init__(
        self,
        lr:              float = 1e-4,
        weight_decay:    float = 1e-3,
        label_smoothing: float = 0.05,
        dropout:         float = 0.15,
        head_hidden:     int   = 512,
        max_epochs:      int   = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str]  = []
        self._test_symbols:  List[str]  = []
        self._test_labels:   List[torch.Tensor] = []

        self._edge_index:  Optional[torch.Tensor] = None
        self._edge_weight: Optional[torch.Tensor] = None
        self._n_nodes: int = 18870

    def setup(self, stage=None):
        dm = self.trainer.datamodule if self.trainer is not None else None
        if dm is not None and hasattr(dm, "edge_index"):
            self._edge_index  = dm.edge_index
            self._edge_weight = dm.edge_weight
            self._n_nodes     = dm.n_nodes

        self.model = StringGNNCondBilinearModel(
            edge_index  = self._edge_index,
            edge_weight = self._edge_weight,
            n_nodes     = self._n_nodes,
            dropout     = self.hparams.dropout,
            head_hidden = self.hparams.head_hidden,
        )
        # Float32 for trainable parameters (stable optimization)
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.register_buffer("class_weights", CLASS_WEIGHTS)

    def forward(self, node_indices):
        return self.model(node_indices)

    def _loss(self, logits, labels):
        # Reshape to [B*6640, 3] / [B*6640] to use nll_loss (1D, deterministic)
        # instead of nll_loss2d which lacks a deterministic CUDA implementation.
        logits_2d = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)  # [B*G, 3]
        labels_1d = labels.reshape(-1)                               # [B*G]
        return F.cross_entropy(
            logits_2d, labels_1d,
            weight=self.class_weights.to(logits.device),
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
        keep_indices: List[int] = []
        for i, pid in enumerate(all_pert):
            if pid not in seen_pids:
                seen_pids.add(pid)
                keep_indices.append(i)
        if len(keep_indices) < len(all_pert):
            self.print(f"[Node4-1] Deduplicating: {len(all_pert)} → {len(keep_indices)}")
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
            self.print(f"[Node4-1] Saved test predictions → {pred_path}")
            if all_labels.any():
                f1 = compute_per_gene_f1(all_probs.numpy(), all_labels.numpy())
                self.print(f"[Node4-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
        )
        # CosineAnnealingLR: smooth decay from lr to eta_min over max_epochs
        # Avoids the step-wise aggressive decay of ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hp.max_epochs, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
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
        description="Node 4-1 – STRING_GNN cond_emb + Multi-Layer Fusion + Enhanced Head"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--lr",                 type=float, default=1e-4)
    p.add_argument("--weight-decay",       type=float, default=1e-3)
    p.add_argument("--label-smoothing",    type=float, default=0.05)
    p.add_argument("--dropout",            type=float, default=0.15)
    p.add_argument("--head-hidden",        type=int,   default=512)
    p.add_argument("--micro-batch-size",   type=int,   default=4)
    p.add_argument("--global-batch-size",  type=int,   default=32)
    p.add_argument("--max-epochs",         type=int,   default=200)
    p.add_argument("--patience",           type=int,   default=25)
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
    lit = StringGNNCondLitModule(
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        label_smoothing = args.label_smoothing,
        dropout         = args.dropout,
        head_hidden     = args.head_hidden,
        max_epochs      = args.max_epochs,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max",
                           patience=args.patience, min_delta=1e-5)
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
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,   # GCNConv scatter_add is deterministic on CUDA
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 4-1 – STRING_GNN cond_emb + Multi-Layer Fusion + Enhanced Head\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
