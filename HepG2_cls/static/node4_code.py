"""Node 4: ESM2-650M + STRING_GNN Gated Cross-Attention Fusion.

Architecture:
  - Branch A — ESM2-650M (protein sequence encoder):
      Protein sequence of perturbed gene → ESM2 mean-pool → [B, 1280]
      Full fine-tuning of ESM2-650M (21GB at seq_len=512, well within H100 80GB)
  - Branch B — STRING_GNN (PPI graph encoder):
      Run STRING_GNN conditioned on perturbed gene → extract node embedding for
      perturbed gene from the graph output → [B, 256]
  - Gated Fusion:
      Project both branches to shared dim D=512
      Gate = sigmoid(W_a * a + W_b * b)
      fused = gate * a_proj + (1 - gate) * b_proj   → [B, 512]
  - Head: 2-layer MLP → [B, 6640 * 3] → reshape to [B, 3, 6640]
  - Loss: Focal loss with class weights

Auxiliary data:
  - ESM2 protein sequences: /home/data/genome/hg38_gencode_protein.fa
  - STRING_GNN graph: /home/Models/STRING_GNN/graph_data.pt
  - STRING_GNN node names: /home/Models/STRING_GNN/node_names.json
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, EsmForMaskedLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
STRING_GNN_DIR = "/home/Models/STRING_GNN"
PROTEIN_FASTA = "/home/data/genome/hg38_gencode_protein.fa"
N_GENES = 6640
N_CLASSES = 3
ESM2_MAX_LEN = 512      # per skill: max 1024; 512 covers most proteins
ESM2_HIDDEN = 1280      # ESM2-650M hidden size
GNN_HIDDEN = 256        # STRING_GNN embedding dim
FUSED_DIM = 512         # dimension after gated fusion


# ---------------------------------------------------------------------------
# Protein FASTA helpers (same as node2)
# ---------------------------------------------------------------------------
def _build_ensg_to_seq(fasta_path: str) -> Dict[str, str]:
    """Build ENSG→longest protein sequence map from GENCODE protein FASTA."""
    ensg2seq: Dict[str, str] = {}
    current_ensg: Optional[str] = None
    current_seq_parts: List[str] = []

    def _flush():
        if current_ensg and current_seq_parts:
            seq = "".join(current_seq_parts)
            if current_ensg not in ensg2seq or len(seq) > len(ensg2seq[current_ensg]):
                ensg2seq[current_ensg] = seq

    with open(fasta_path, "r") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                _flush()
                current_seq_parts = []
                current_ensg = None
                fields = line[1:].split("|")
                if len(fields) >= 3:
                    current_ensg = fields[2].split(".")[0]
            else:
                current_seq_parts.append(line)
    _flush()
    return ensg2seq


_ENSG2SEQ_CACHE: Optional[Dict[str, str]] = None

def get_ensg2seq() -> Dict[str, str]:
    global _ENSG2SEQ_CACHE
    if _ENSG2SEQ_CACHE is None:
        _ENSG2SEQ_CACHE = _build_ensg_to_seq(PROTEIN_FASTA)
    return _ENSG2SEQ_CACHE


FALLBACK_SEQ = "M"


# ---------------------------------------------------------------------------
# STRING_GNN helpers
# ---------------------------------------------------------------------------
def _build_ensg_to_node_idx(node_names_path: str) -> Dict[str, int]:
    """Map ENSG IDs (no version) to STRING_GNN node indices."""
    with open(node_names_path, "r") as f:
        node_names: List[str] = json.load(f)
    ensg2idx = {}
    for i, name in enumerate(node_names):
        ensg = name.split(".")[0]  # strip version if any
        ensg2idx[ensg] = i
    return ensg2idx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        ensg2seq: Dict[str, str],
        ensg2node: Dict[str, int],
        n_gnn_nodes: int,
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.n_gnn_nodes = n_gnn_nodes

        # Resolve protein sequences
        self.sequences: List[str] = []
        for pid in self.pert_ids:
            ensg = pid.split(".")[0]
            self.sequences.append(ensg2seq.get(ensg, FALLBACK_SEQ))

        # Resolve STRING_GNN node indices (-1 means not in PPI graph)
        self.node_indices: List[int] = []
        for pid in self.pert_ids:
            ensg = pid.split(".")[0]
            self.node_indices.append(ensg2node.get(ensg, -1))

        if "label" in df.columns:
            labels = np.array([json.loads(x) for x in df["label"].tolist()], dtype=np.int64)
            self.labels: Optional[torch.Tensor] = torch.tensor(labels + 1, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "seq": self.sequences[idx],
            "node_idx": self.node_indices[idx],  # int, -1 if not in graph
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class PerturbDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        micro_batch_size: int = 2,
        num_workers: int = 4,
        max_seq_len: int = ESM2_MAX_LEN,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len

        self.tokenizer = None
        self.ensg2seq: Optional[Dict[str, str]] = None
        self.ensg2node: Optional[Dict[str, int]] = None
        self.n_gnn_nodes: int = 18870
        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: str = "fit") -> None:
        # ESM2 tokenizer
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(ESM2_MODEL)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)

        self.ensg2seq = get_ensg2seq()
        self.ensg2node = _build_ensg_to_node_idx(
            str(Path(STRING_GNN_DIR) / "node_names.json")
        )
        self.n_gnn_nodes = len(self.ensg2node)

        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbDataset(train_df, self.ensg2seq, self.ensg2node, self.n_gnn_nodes)
        self.val_ds = PerturbDataset(val_df, self.ensg2seq, self.ensg2node, self.n_gnn_nodes)
        self.test_ds = PerturbDataset(test_df, self.ensg2seq, self.ensg2node, self.n_gnn_nodes)

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        seqs = [item["seq"] for item in batch]
        tokenized = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        result = {
            "idx": torch.tensor([item["idx"] for item in batch], dtype=torch.long),
            "pert_id": [item["pert_id"] for item in batch],
            "symbol": [item["symbol"] for item in batch],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            # node indices for STRING_GNN lookup: [B], -1 = not in graph
            "node_idx": torch.tensor([item["node_idx"] for item in batch], dtype=torch.long),
        }
        if "label" in batch[0]:
            result["label"] = torch.stack([item["label"] for item in batch])
        return result

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )


# ---------------------------------------------------------------------------
# Gated Fusion
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Learnable gated fusion of two embeddings.

    gate = sigmoid(W_a * a + W_b * b + bias)
    fused = gate * a_proj + (1 - gate) * b_proj
    """

    def __init__(self, dim_a: int, dim_b: int, out_dim: int) -> None:
        super().__init__()
        self.proj_a = nn.Linear(dim_a, out_dim)
        self.proj_b = nn.Linear(dim_b, out_dim)
        self.gate_a = nn.Linear(dim_a, out_dim, bias=False)
        self.gate_b = nn.Linear(dim_b, out_dim, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(out_dim))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: [B, dim_a], b: [B, dim_b]
        a_proj = self.proj_a(a)   # [B, out_dim]
        b_proj = self.proj_b(b)   # [B, out_dim]
        gate = torch.sigmoid(
            self.gate_a(a) + self.gate_b(b) + self.gate_bias
        )  # [B, out_dim]
        fused = gate * a_proj + (1.0 - gate) * b_proj
        return self.norm(fused)   # [B, out_dim]


# ---------------------------------------------------------------------------
# Prediction Head
# ---------------------------------------------------------------------------
class PerturbHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 1024, n_genes: int = N_GENES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_genes * N_CLASSES),
        )
        self.n_genes = n_genes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, N_CLASSES, self.n_genes)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(input, dim=-1)
        prob = torch.exp(log_prob)
        log_p_t = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = prob.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_w = (1 - p_t) ** self.gamma
        loss = -focal_w * log_p_t
        if self.weight is not None:
            w = self.weight.to(device=input.device, non_blocking=True)
            loss = loss * w[target]
        return loss.mean()


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        head_hidden_dim: int = 1024,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.esm2: Optional[EsmForMaskedLM] = None
        self.string_gnn = None
        self.fusion: Optional[GatedFusion] = None
        self.head: Optional[PerturbHead] = None

        # Will hold the fixed STRING graph buffers (registered in setup via register_buffer)
        # Cached GNN node embeddings (recomputed per forward during training, cached during eval)
        self._gnn_node_emb_cache: Optional[torch.Tensor] = None

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # ESM2-650M in float32 for full fine-tuning (5.05 GB/GPU per memory table)
        self.esm2 = EsmForMaskedLM.from_pretrained(ESM2_MODEL, dtype=torch.float32)
        self.esm2.gradient_checkpointing_enable()

        # STRING_GNN (small: 5.43M params, full fine-tune)
        self.string_gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)

        # Load graph data
        graph = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", map_location="cpu")
        # Register as buffers so they move with the model
        self.register_buffer("_graph_edge_index", graph["edge_index"].long())
        self.register_buffer("_graph_edge_weight", graph["edge_weight"].float())

        # Gated fusion and prediction head
        self.fusion = GatedFusion(dim_a=ESM2_HIDDEN, dim_b=GNN_HIDDEN, out_dim=FUSED_DIM)
        self.head = PerturbHead(in_dim=FUSED_DIM, hidden_dim=self.hparams.head_hidden_dim)

        # Class weights for focal loss
        freq = torch.tensor([0.9282, 0.0477, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq)
        class_weights = class_weights / class_weights.mean()
        self.focal_loss = FocalLoss(gamma=self.hparams.focal_gamma, weight=class_weights)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"ESM2-650M + STRING_GNN | total={total:,} | trainable={trainable:,}")

        # Cast all trainable parameters to float32 for stable optimization
        for p in self.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    def _get_esm2_emb(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """ESM2 mean pool over residues (excluding CLS, EOS, PAD).

        Returns: [B, 1280]
        """
        out = self.esm2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = out["hidden_states"][-1]  # [B, T, 1280]

        # Exclude special tokens: CLS (0), EOS (2), PAD (1)
        tokenizer_special_ids = torch.tensor(
            [0, 1, 2], dtype=torch.long, device=input_ids.device
        )
        special_mask = torch.isin(input_ids, tokenizer_special_ids)  # [B, T]
        valid_mask = ~special_mask  # [B, T]
        valid_mask_f = valid_mask.unsqueeze(-1).float()  # [B, T, 1]
        sum_emb = (hidden * valid_mask_f).sum(dim=1)     # [B, 1280]
        count = valid_mask_f.sum(dim=1).clamp(min=1e-9)  # [B, 1]
        return sum_emb / count  # [B, 1280]

    def _get_gnn_emb(self, node_idx: torch.Tensor) -> torch.Tensor:
        """Run STRING_GNN and extract node embeddings for perturbed genes.

        Args:
            node_idx: [B] int — STRING_GNN node indices; -1 means not in graph

        Returns: [B, 256]
        """
        device = node_idx.device
        out = self.string_gnn(
            edge_index=self._graph_edge_index,
            edge_weight=self._graph_edge_weight,
        )
        all_emb = out.last_hidden_state  # [N_nodes, 256]

        # Gather embeddings for each sample; fallback to zeros for unknown genes
        B = node_idx.size(0)
        result = torch.zeros(B, GNN_HIDDEN, dtype=all_emb.dtype, device=device)
        valid = node_idx >= 0
        if valid.any():
            result[valid] = all_emb[node_idx[valid]]
        return result.float()

    def _forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        esm2_emb = self._get_esm2_emb(batch["input_ids"], batch["attention_mask"])  # [B, 1280]
        gnn_emb = self._get_gnn_emb(batch["node_idx"])                              # [B, 256]
        fused = self.fusion(esm2_emb, gnn_emb)                                      # [B, 512]
        return self.head(fused)                                                       # [B, 3, 6640]

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return self.focal_loss(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self._forward(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self._forward(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds, dim=0).numpy()
        labels = torch.cat(self._val_labels, dim=0).numpy()
        self._val_preds.clear()
        self._val_labels.clear()
        f1 = _compute_per_gene_f1(preds, labels)
        self.log("val/f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self._forward(batch)
        self._test_preds.append(logits.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        preds_local = torch.cat(self._test_preds, dim=0)
        self._test_preds.clear()

        all_preds = self.all_gather(preds_local)

        # Gather pert_ids and symbols from ALL ranks via torch.distributed
        all_pert_ids: List[str] = []
        all_symbols: List[str] = []
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            _gathered_ids: List[List[str]] = [None] * self.trainer.world_size
            _gathered_syms: List[List[str]] = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(_gathered_ids, self._test_pert_ids)
            torch.distributed.all_gather_object(_gathered_syms, self._test_symbols)
            for ids in _gathered_ids:
                all_pert_ids.extend(ids)
            for syms in _gathered_syms:
                all_symbols.extend(syms)
        else:
            all_pert_ids = self._test_pert_ids[:]
            all_symbols = self._test_symbols[:]
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        if self.trainer is not None and self.trainer.world_size > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)

        if self.trainer is not None and self.trainer.is_global_zero:
            n = min(len(all_pert_ids), len(all_preds))
            _save_test_predictions(
                pert_ids=all_pert_ids[:n],
                symbols=all_symbols[:n],
                preds=all_preds.float().cpu().numpy()[:n],
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

    def configure_optimizers(self):
        # ESM2 backbone gets lower lr; fusion + head get higher lr
        esm2_params = list(self.esm2.parameters())
        esm2_param_ids = {id(p) for p in esm2_params}
        other_params = [p for p in self.parameters() if id(p) not in esm2_param_ids]

        optimizer = torch.optim.AdamW(
            [
                {"params": esm2_params, "lr": self.hparams.lr * 0.1},    # backbone LR
                {"params": other_params, "lr": self.hparams.lr},          # head/gnn/fusion LR
            ],
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/f1", "interval": "epoch"},
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        result = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    result[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                result[key] = full_sd[key]
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving {trainable}/{total} params ({100*trainable/total:.2f}%)")
        return result

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import f1_score as sk_f1
    y_hat = preds.argmax(axis=1)
    n_genes = labels.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels[:, g]
        yh = y_hat[:, g]
        per_class_f1 = sk_f1(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        f1_vals.append(float(per_class_f1[present].mean()))
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    n = min(len(pert_ids), len(preds))
    for i in range(n):
        rows.append({
            "idx": pert_ids[i],
            "input": symbols[i],
            "prediction": json.dumps(preds[i].tolist()),
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Node4: ESM2-650M + STRING_GNN Fusion for HepG2 DEG")
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--global-batch-size", type=int, default=16)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--head-hidden-dim", type=int, default=1024)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--val-check-interval", type=float, default=1.0,
                   help="Validation check interval as fraction of training set (default: 1.0 = once per epoch)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        max_seq_len=ESM2_MAX_LEN,
    )

    model = PerturbModule(
        head_hidden_dim=args.head_hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
    )

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = limit_val = limit_test = args.debug_max_step
        max_steps = args.debug_max_step
        val_check_interval = 1.0
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1
        val_check_interval = args.val_check_interval

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val_f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(monitor="val/f1", mode="max", patience=args.early_stop_patience, min_delta=1e-5)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule)

    # Resolve the best checkpoint path: use the one written by the ModelCheckpoint
    # whose dirpath=output_dir/checkpoints and filename=best-{epoch}-{val_f1:.4f}.
    # All ranks save to the same location via default_root_dir in ModelCheckpoint.
    ckpt_dir = output_dir / "checkpoints"
    best_ckpt = None
    if ckpt_dir.exists():
        candidates = sorted(ckpt_dir.glob("best-*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for c in candidates:
            if c.is_file() and c.suffix == ".ckpt":
                best_ckpt = str(c)
                break
            # directory-style checkpoint: best-epoch=018-val_f1=0.5005.ckpt/
            sub_ckpt = next((f for f in c.iterdir() if f.suffix == ".ckpt"), None)
            if sub_ckpt:
                best_ckpt = str(sub_ckpt)
                break

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path=best_ckpt)

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results → {score_path}")


if __name__ == "__main__":
    main()
