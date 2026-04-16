"""Node 3-1-2: STRING_GNN + AIDO.Cell-10M Hybrid Fusion for K562 DEG Prediction.

This node implements a hybrid architecture combining:
1. Frozen STRING_GNN (PPI topology, 256-dim per gene) - proven to reach 0.485+ F1
2. AIDO.Cell-10M with QKV-only fine-tuning (Muon) - captures transcriptome context
3. Concatenation fusion (1280-dim total: 1024-dim AIDO.Cell + 256-dim STRING_GNN)
4. Label-smoothed CE + class weights (proven in node3 parent and siblings)
5. SGDR (T_0=15, T_mult=2) - proven in node3-3-1-1 (best AIDO.Cell result: F1=0.4368)

Key differences from sibling node3-1-1:
- Adds frozen STRING_GNN backbone for PPI topology signal
- Uses SGDR instead of CosineAnnealingWarmRestarts(T_0=50)
- Total input to head: 1280-dim (vs 1024-dim in sibling)
- Target: break the ~0.43 AIDO.Cell-only ceiling by adding STRING_GNN's PPI signal
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES    = 6640
N_CLASSES  = 3
AIDO_GENES = 19264
AIDO_MODEL_DIR    = "/home/Models/AIDO.Cell-10M"
STRING_GNN_DIR    = "/home/Models/STRING_GNN"
HIDDEN_DIM = 256      # AIDO.Cell-10M hidden size
N_LAYERS   = 8        # AIDO.Cell-10M transformer layers
STRING_DIM = 256      # STRING_GNN embedding dimension

CLASS_FREQ = [0.0429, 0.9251, 0.0320]  # down, neutral, up (remapped 0,1,2)

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency class weights for class imbalance."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute per-gene macro F1, matching the calc_metric.py evaluation logic."""
    y_hat       = preds.argmax(dim=1)
    G           = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)
    for c in range(3):
        is_true = (targets == c)
        is_pred = (y_hat == c)
        present = is_true.any(dim=0).float()
        tp  = (is_pred & is_true).float().sum(0)
        fp  = (is_pred & ~is_true).float().sum(0)
        fn  = (~is_pred & is_true).float().sum(0)
        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(prec + rec > 0, 2*prec*rec/(prec+rec+1e-8), torch.zeros_like(prec))
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


def load_string_gnn_embeddings() -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load frozen STRING_GNN embeddings for all nodes.

    Returns:
        emb_matrix: [18870, 256] float32 tensor with per-gene PPI embeddings
        name_to_idx: dict mapping Ensembl gene ID -> row index in emb_matrix
    """
    import json
    node_names = json.loads((Path(STRING_GNN_DIR) / "node_names.json").read_text())
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    # Load STRING_GNN and compute embeddings (run once, frozen)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_model = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True).to(device)
    gnn_model.eval()

    graph = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", map_location=device)
    edge_index  = graph["edge_index"]
    edge_weight = graph.get("edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    with torch.no_grad():
        outputs = gnn_model(edge_index=edge_index, edge_weight=edge_weight)
        emb_matrix = outputs.last_hidden_state.float().cpu()  # [18870, 256]

    del gnn_model
    torch.cuda.empty_cache()

    return emb_matrix, name_to_idx


# ---------------------------------------------------------------------------
# Dataset / DataModule
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame, string_emb: torch.Tensor,
                 name_to_idx: Dict[str, int]) -> None:
        self.pert_ids   = df["pert_id"].tolist()
        self.symbols    = df["symbol"].tolist()
        self.string_emb = string_emb      # [18870, 256]
        self.name_to_idx = name_to_idx

        has_label = "label" in df.columns and df["label"].notna().all()
        self.labels: Optional[List] = (
            [torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
             for row in df["label"].tolist()]
            if has_label else None
        )

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid = self.pert_ids[idx]
        # Look up STRING_GNN embedding for the perturbed gene
        gnn_idx = self.name_to_idx.get(pid, -1)
        if gnn_idx >= 0:
            gnn_emb = self.string_emb[gnn_idx]   # [256]
        else:
            gnn_emb = torch.zeros(STRING_DIM)     # fallback for OOV genes

        item: Dict[str, Any] = {
            "sample_idx": idx,
            "pert_id":    pid,
            "symbol":     self.symbols[idx],
            "gnn_emb":    gnn_emb,
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def make_collate(tokenizer):
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        input_ids  = tokenized["input_ids"]  # [B, 19264] float32
        gene_in_vocab  = (input_ids > -1.0).any(dim=1)
        gene_positions = torch.where(
            gene_in_vocab,
            (input_ids > -1.0).float().argmax(dim=1),
            torch.zeros(len(batch), dtype=torch.long),
        )
        out: Dict[str, Any] = {
            "sample_idx":     torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":        pert_ids,
            "symbol":         symbols,
            "input_ids":      input_ids,
            "attention_mask": tokenized["attention_mask"],
            "gene_positions": gene_positions,
            "gnn_emb":        torch.stack([b["gnn_emb"] for b in batch]),  # [B, 256]
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out
    return collate_fn


class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.tokenizer   = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Load tokenizer (rank-0 first)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        # Load frozen STRING_GNN embeddings once
        string_emb, name_to_idx = load_string_gnn_embeddings()

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")
        self.train_ds = DEGDataset(train_df, string_emb, name_to_idx)
        self.val_ds   = DEGDataset(val_df,   string_emb, name_to_idx)
        self.test_ds  = DEGDataset(test_df,  string_emb, name_to_idx)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=make_collate(self.tokenizer))

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class HybridStringGNNAIDOModel(pl.LightningModule):
    """Hybrid STRING_GNN + AIDO.Cell-10M model for K562 DEG prediction.

    Architecture:
    - Frozen STRING_GNN embeddings (256-dim, pre-computed from PPI graph)
    - AIDO.Cell-10M with QKV-only fine-tuning using Muon optimizer
    - Concat fusion: [AIDO 4-layer concat (1024-dim)] + [STRING_GNN (256-dim)] = 1280-dim
    - MLP head: Linear(1280→512) + LN + GELU + Dropout(0.3) + Linear(512→N_CLASSES*N_GENES)
    - Label-smoothed CE + sqrt-inverse-frequency class weights
    - SGDR (CosineAnnealingWarmRestarts T_0=15, T_mult=2) - proven in node3-3-1-1

    Key innovations:
    1. STRING_GNN provides PPI topology for the perturbed gene (which genes interact with it)
    2. AIDO.Cell provides transcriptomic context (how genes co-express)
    3. Combined representation is richer than either alone
    """

    def __init__(
        self,
        fusion_layers: int  = 4,       # last N AIDO.Cell transformer layers to concatenate
        head_hidden: int    = 512,     # MLP head hidden dim
        head_dropout: float = 0.3,     # dropout in head
        lr_muon: float      = 0.02,    # Muon LR for QKV weight matrices
        lr_adamw: float     = 2e-4,    # AdamW LR for head + non-QKV params
        weight_decay: float = 2e-2,
        sgdr_t0: int        = 15,      # SGDR T_0 (proven in node3-3-1-1)
        sgdr_t_mult: int    = 2,       # SGDR T_mult
        sgdr_min_lr_ratio: float = 0.05,  # minimum LR as fraction of base
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load AIDO.Cell-10M backbone ----
        self.aido_backbone = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        self.aido_backbone = self.aido_backbone.to(torch.bfloat16)
        self.aido_backbone.config.use_cache = False
        self.aido_backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Enable FlashAttention to avoid OOM from full 19266x19266 attention matrix
        self.aido_backbone.config._use_flash_attention_2 = True

        # Share QKV weight tensors between flash_self and self.self
        for layer in self.aido_backbone.bert.encoder.layer:
            ss = layer.attention.flash_self  # BertSelfFlashAttention
            mm = layer.attention.self       # CellFoundationSelfAttention (regular)
            ss.query.weight = mm.query.weight
            ss.key.weight   = mm.key.weight
            ss.value.weight = mm.value.weight
            ss.query.bias   = mm.query.bias
            ss.key.bias     = mm.key.bias
            ss.value.bias   = mm.value.bias

        # Freeze all AIDO.Cell layers, then unfreeze only QKV weights
        for param in self.aido_backbone.parameters():
            param.requires_grad = False

        qkv_patterns = (
            "attention.self.query.weight",
            "attention.self.key.weight",
            "attention.self.value.weight",
        )
        for name, param in self.aido_backbone.named_parameters():
            if any(name.endswith(p) for p in qkv_patterns):
                param.requires_grad = True

        qkv_count = sum(p.numel() for p in self.aido_backbone.parameters() if p.requires_grad)
        total      = sum(p.numel() for p in self.aido_backbone.parameters())
        print(f"[Node3-1-2] AIDO trainable QKV params: {qkv_count:,} / {total:,}")

        # ---- Head: concat(AIDO 4-layer, STRING_GNN) → classification ----
        # Input: 4 * HIDDEN_DIM (AIDO concat) + STRING_DIM (GNN)
        # = 4 * 256 + 256 = 1280-dim
        aido_fused_dim = hp.fusion_layers * HIDDEN_DIM  # 4 * 256 = 1024
        total_fused_dim = aido_fused_dim + STRING_DIM   # 1024 + 256 = 1280

        self.head = nn.Sequential(
            nn.Linear(total_fused_dim, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )

        # Cast head parameters to float32 for stable optimization
        # Note: must cast individual parameters, NOT reassign self.head
        for p in self.head.parameters():
            p.data = p.data.float()

        # ---- Loss with class weights ----
        class_weights = get_class_weights()
        self.register_buffer("class_weights", class_weights)

        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds: List[torch.Tensor] = []
        self._test_meta:  List[Tuple]        = []

    # ---- forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_positions: torch.Tensor,
        gnn_emb: torch.Tensor,
    ) -> torch.Tensor:
        B = input_ids.shape[0]

        # AIDO.Cell forward pass with hidden states
        out = self.aido_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # hidden_states: tuple of (N_LAYERS+1) tensors, each [B, AIDO_GENES+2, 256]
        hidden_states = out.hidden_states  # len = N_LAYERS + 1 = 9

        n = self.hparams.fusion_layers
        # Collect per-layer embeddings at the perturbed gene position (last 4 layers)
        layer_embs = []
        for i in range(n):
            hs = hidden_states[-(i + 1)]   # [B, AIDO_GENES+2, 256]
            ge = hs[torch.arange(B, device=hs.device), gene_positions, :].float()  # [B, 256]
            layer_embs.append(ge)

        # Concatenate last 4 layers: [B, 4*256=1024]
        aido_features = torch.cat(layer_embs, dim=-1)

        # STRING_GNN embedding is pre-computed [B, 256], already float
        gnn_features = gnn_emb.float().to(aido_features.device)

        # Fused representation: [B, 1280]
        fused = torch.cat([aido_features, gnn_features], dim=-1)

        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        return logits

    # ---- loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        flat_logits  = logits.permute(0, 2, 1).reshape(-1, C)
        flat_targets = targets.reshape(-1)
        return F.cross_entropy(
            flat_logits, flat_targets,
            weight=self.class_weights.to(flat_logits.device),
            label_smoothing=0.1,
        )

    # ---- steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"],
                      batch["gene_positions"], batch["gnn_emb"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"],
                      batch["gene_positions"], batch["gnn_emb"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits.float(), dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, 0)
        local_tgts  = torch.cat(self._val_tgts,  0)
        local_idx   = torch.cat(self._val_idx,   0)
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"],
                      batch["gene_positions"], batch["gnn_emb"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        for i, (pid, sym) in enumerate(zip(batch["pert_id"], batch["symbol"])):
            self._test_meta.append((pid, sym, batch["sample_idx"][i].item()))
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds = torch.cat(self._test_preds, 0)   # [local_N, 3, 6640]
        local_idx  = torch.tensor(
            [m[2] for m in self._test_meta], dtype=torch.long, device=local_preds.device
        )

        # Build (sample_idx, pert_id, symbol, predictions) list per rank
        # This avoids the need to align predictions + meta separately
        local_rows = []
        for i, (pid, sym, sidx) in enumerate(self._test_meta):
            local_rows.append({
                "sample_idx": sidx,
                "pert_id":    pid,
                "symbol":     sym,
                "prediction": local_preds[i].cpu().numpy().tolist(),
            })

        # Collect predictions and meta on rank 0 via all_gather_object.
        # Both branches call the collective together so there's no mismatch.
        if self.trainer.is_global_zero:
            if torch.distributed.is_initialized():
                world_size   = torch.distributed.get_world_size()
                all_meta_obj = [None] * world_size
                torch.distributed.all_gather_object(all_meta_obj, local_rows)
            else:
                all_meta_obj = [local_rows]
        else:
            if torch.distributed.is_initialized():
                _dummy = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(_dummy, local_rows)

        self._test_preds.clear()
        self._test_meta.clear()

        if not self.trainer.is_global_zero:
            return

        # Flatten meta from all ranks
        global_rows = []
        for rank_rows in all_meta_obj:
            global_rows.extend(rank_rows)

        # Deduplicate by sample_idx (keep first occurrence)
        seen = set()
        unique_rows = []
        for row in global_rows:
            if row["sample_idx"] not in seen:
                seen.add(row["sample_idx"])
                unique_rows.append(row)

        rows = []
        for row in unique_rows:
            rows.append({
                "idx":        row["pert_id"],
                "input":      row["symbol"],
                "prediction": json.dumps(row["prediction"]),
            })
        out_dir = Path(__file__).parent / "run"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
        print(f"[Node3-1-2] Saved {len(rows)} test predictions.")

    # ---- checkpoint: save only trainable parameters ----
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                key = prefix + name
                if key in full:
                    trainable[key] = full[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full:
                trainable[key] = full[key]
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Checkpoint: {trained}/{total} params ({100*trained/total:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- optimizer: Muon for QKV weight matrices, AdamW for everything else ----
    def configure_optimizers(self):
        hp = self.hparams

        # QKV weight matrices for Muon (ndim >= 2, square matrices)
        qkv_weights = [
            p for name, p in self.aido_backbone.named_parameters()
            if p.requires_grad and p.ndim >= 2
        ]
        # Head parameters for AdamW
        head_params = list(self.head.parameters())

        param_groups = [
            # Muon group: QKV weight matrices
            dict(
                params       = qkv_weights,
                use_muon     = True,
                lr           = hp.lr_muon,
                weight_decay = hp.weight_decay,
                momentum     = 0.95,
            ),
            # AdamW group: head + non-QKV backbone params
            dict(
                params       = head_params,
                use_muon     = False,
                lr           = hp.lr_adamw,
                betas        = (0.9, 0.95),
                weight_decay = hp.weight_decay,
            ),
        ]
        use_distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        opt_cls   = MuonWithAuxAdam if use_distributed else SingleDeviceMuonWithAuxAdam
        optimizer = opt_cls(param_groups)

        # SGDR: CosineAnnealingWarmRestarts with T_0=15, T_mult=2
        # Proven effective in node3-3-1-1 (best AIDO.Cell result: F1=0.4368)
        # With T_0=15 and T_mult=2: restarts at epoch 15, 45, 105, ...
        # The epoch-15 restart aligns well with typical peak performance epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0     = hp.sgdr_t0,
            T_mult  = hp.sgdr_t_mult,
            eta_min = hp.lr_muon * hp.sgdr_min_lr_ratio,
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node3-1-2: STRING_GNN + AIDO.Cell-10M Hybrid for K562 DEG Prediction"
    )
    parser.add_argument("--micro-batch-size",  type=int,   default=16)
    parser.add_argument("--global-batch-size", type=int,   default=128)
    parser.add_argument("--max-epochs",        type=int,   default=120)
    parser.add_argument("--lr-muon",           type=float, default=0.02)
    parser.add_argument("--lr-adamw",          type=float, default=2e-4)
    parser.add_argument("--weight-decay",      type=float, default=2e-2)
    parser.add_argument("--fusion-layers",     type=int,   default=4)
    parser.add_argument("--head-hidden",       type=int,   default=512)
    parser.add_argument("--head-dropout",      type=float, default=0.3)
    parser.add_argument("--sgdr-t0",           type=int,   default=15)
    parser.add_argument("--sgdr-t-mult",       type=int,   default=2)
    parser.add_argument("--sgdr-min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--debug_max_step",    type=int,   default=None)
    parser.add_argument("--fast_dev_run",      action="store_true")
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = lim_val = lim_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = lim_val = lim_test = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = HybridStringGNNAIDOModel(
        fusion_layers        = args.fusion_layers,
        head_hidden          = args.head_hidden,
        head_dropout         = args.head_dropout,
        lr_muon              = args.lr_muon,
        lr_adamw             = args.lr_adamw,
        weight_decay         = args.weight_decay,
        sgdr_t0              = args.sgdr_t0,
        sgdr_t_mult          = args.sgdr_t_mult,
        sgdr_min_lr_ratio    = args.sgdr_min_lr_ratio,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1", mode="max", save_top_k=1,
    )
    # Early stopping: patience=10 (balanced), min_delta=0.002
    # Prevents post-peak overfitting while allowing SGDR warm restarts to recover
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=10, min_delta=0.002)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    strategy = (
        DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator             = "gpu",
        devices                 = n_gpus,
        num_nodes               = 1,
        strategy                = strategy,
        precision               = "bf16-mixed",
        max_epochs              = args.max_epochs,
        max_steps               = max_steps,
        accumulate_grad_batches = accum,
        limit_train_batches     = lim_train,
        limit_val_batches       = lim_val,
        limit_test_batches      = lim_test,
        val_check_interval      = args.val_check_interval if (args.debug_max_step is None and not fast_dev_run) else 1.0,
        num_sanity_val_steps    = 2,
        callbacks               = [ckpt_cb, es_cb, lr_cb, pg_cb],
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = 1.0,
    )

    trainer.fit(model, datamodule=dm)

    ckpt_path = "best" if (args.debug_max_step is None and not fast_dev_run) else None
    test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)

    # Compute real F1 from saved test predictions using calc_metric.py
    score_path = Path(__file__).parent / "test_score.txt"
    pred_path  = Path(__file__).parent / "run" / "test_predictions.tsv"
    if pred_path.exists() and Path(TEST_TSV).exists():
        import subprocess
        try:
            result = subprocess.run(
                ["python", str(DATA_ROOT / "calc_metric.py"), str(pred_path), str(TEST_TSV)],
                capture_output=True, text=True, timeout=120
            )
            metrics = json.loads(result.stdout.strip().split("\n")[-1])
            f1_score = metrics.get("value", None)
            if f1_score is not None:
                with open(score_path, "w") as f:
                    f.write(f"f1_score: {f1_score}\n")
                    if "details" in metrics:
                        for k, v in metrics["details"].items():
                            f.write(f"  {k}: {v}\n")
                print(f"[Node3-1-2] test_f1={f1_score:.4f} — saved to {score_path}")
            else:
                with open(score_path, "w") as f:
                    f.write(f"error: {metrics.get('error', 'unknown')}\n")
        except Exception as e:
            with open(score_path, "w") as f:
                f.write(f"error: {e}\n")
            print(f"[Node3-1-2] calc_metric failed: {e}")
    else:
        with open(score_path, "w") as f:
            f.write(f"error: test_predictions.tsv not found\n")
    print(f"[Node3-1-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
