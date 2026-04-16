"""
Node 1-1-3-1-1 – Triple-Stream Hybrid: AIDO.Cell-100M + STRING_GNN PPI Embeddings

The key architectural innovation is adding frozen precomputed STRING_GNN PPI embeddings
as a third stream alongside AIDO.Cell's dual streams, forming a triple-stream fusion:

  Stream A: AIDO.Cell gene-specific position embedding  [B, 640]
  Stream B: AIDO.Cell summary token mean               [B, 640]
  Stream C: STRING_GNN PPI embedding (frozen buffer)   [B, 256]
  ─────────────────────────────────────────────────────────────
  Fusion:   LayerNorm(1536) → Linear(1536, 640) → GELU  [B, 640]
  Head:     bilinear → logits                           [B, 3, 6640]

Motivation: the parent node (node1-1-3-1, F1=0.4379) is bottlenecked by AIDO.Cell's
synthetic fixed-baseline representation, which provides minimal cross-perturbation
diversity. STRING_GNN encodes the human PPI graph topology (STRING v12, threshold=900,
18,870 nodes, 256-dim) which directly captures how perturbation signals propagate
through the protein interaction network — the key biological signal for predicting
which downstream genes are up/down-regulated.

Changes from parent (node1-1-3-1):
1. [MAJOR] STRING_GNN PPI embedding as 3rd frozen stream (precomputed once in setup)
2. LoRA r=16, alpha=32 (reverted from r=32/64 — better calibrated for 1,416 samples)
3. global_batch_size=32 (reverted from 64 — 2x more optimizer steps per epoch)
4. warmup_steps=100 (reduced from 150 to match smaller LoRA rank)
5. All other settings retained: lr_backbone=3e-5, lr_head=5e-4, patience=25,
   class_weights=[2.5,1.0,5.5], label_smoothing=0.05, head_dropout=0.15, etc.
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
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ─── Constants ────────────────────────────────────────────────────────────────
N_GENES_OUT = 6640
N_CLASSES = 3
AIDO_MODEL_DIR = "/home/Models/AIDO.Cell-100M"
STRING_GNN_DIR = "/home/Models/STRING_GNN"
AIDO_HIDDEN_DIM = 640
AIDO_N_GENES = 19264
GNN_DIM = 256  # STRING_GNN hidden dimension

# Class weights: down=2.5, neutral=1.0, up=5.5 (unchanged from parent)
CLASS_WEIGHTS = torch.tensor([2.5, 1.0, 5.5], dtype=torch.float32)
PERTURB_EXPRESSION = 10.0
BASELINE_EXPRESSION = 1.0


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


# ─── STRING_GNN Precomputation ────────────────────────────────────────────────
def precompute_string_gnn_embeddings():
    """
    Load pretrained STRING_GNN, run one forward pass on the fixed PPI graph,
    and return the frozen node embedding matrix.

    Returns:
        gnn_emb   : FloatTensor [18870, 256] — one row per STRING node
        node_names: list of Ensembl gene IDs aligned with row indices
    """
    model_dir = Path(STRING_GNN_DIR)
    string_gnn = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    string_gnn.eval()

    graph = torch.load(str(model_dir / "graph_data.pt"), map_location="cpu")
    node_names = json.loads((model_dir / "node_names.json").read_text())

    edge_index = graph["edge_index"]          # [2, E]
    edge_weight = graph.get("edge_weight", None)  # [E] or None

    with torch.no_grad():
        outputs = string_gnn(edge_index=edge_index, edge_weight=edge_weight)
        gnn_emb = outputs.last_hidden_state.float().cpu()  # [18870, 256]

    del string_gnn  # free model memory
    return gnn_emb, node_names


# ─── Dataset & DataModule ─────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    def __init__(self, df, gene_to_pos, all_gene_names, gnn_idx_map, has_labels=True):
        self.pert_ids = df["pert_id"].tolist()
        self.symbols = df["symbol"].tolist()
        # AIDO.Cell vocabulary position (by gene symbol)
        self.gene_positions = [gene_to_pos.get(sym, -1) for sym in self.symbols]
        # STRING_GNN node index (by Ensembl gene ID)
        self.gnn_indices = [gnn_idx_map.get(pid, -1) for pid in self.pert_ids]
        self.all_gene_names = all_gene_names
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
            "gene_pos": self.gene_positions[idx],
            "gnn_idx": self.gnn_indices[idx],
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


def build_collate_fn(tokenizer, all_gene_names):
    base_expression = [BASELINE_EXPRESSION] * len(all_gene_names)

    def collate_fn(batch):
        expr_inputs = []
        for item in batch:
            expr = list(base_expression)
            if item["gene_pos"] >= 0:
                expr[item["gene_pos"]] = PERTURB_EXPRESSION
            expr_inputs.append({"gene_names": all_gene_names, "expression": expr})

        tokenized = tokenizer(expr_inputs, return_tensors="pt")
        gene_positions = torch.tensor([item["gene_pos"] for item in batch], dtype=torch.long)
        gnn_indices = torch.tensor([item["gnn_idx"] for item in batch], dtype=torch.long)

        result = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "gene_pos": gene_positions,
            "gnn_idx": gnn_indices,
            "pert_id": [item["pert_id"] for item in batch],
            "symbol": [item["symbol"] for item in batch],
        }
        if "label" in batch[0]:
            result["label"] = torch.stack([item["label"] for item in batch])
        return result

    return collate_fn


class PerturbationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data", micro_batch_size=4, num_workers=4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        # Rank-0 downloads tokenizer first, then all ranks load
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        gene_to_pos = {sym: pos for sym, pos in tokenizer.gene_to_index.items()}
        all_gene_names = [""] * len(gene_to_pos)
        for sym, pos in gene_to_pos.items():
            all_gene_names[pos] = sym

        # STRING_GNN Ensembl ID → node index map (read-only, fast)
        node_names = json.loads((Path(STRING_GNN_DIR) / "node_names.json").read_text())
        gnn_idx_map = {name: i for i, name in enumerate(node_names)}

        dfs = {split: pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")
               for split in ("train", "val", "test")}

        self.train_ds = PerturbationDataset(
            dfs["train"], gene_to_pos, all_gene_names, gnn_idx_map, True)
        self.val_ds = PerturbationDataset(
            dfs["val"], gene_to_pos, all_gene_names, gnn_idx_map, True)
        self.test_ds = PerturbationDataset(
            dfs["test"], gene_to_pos, all_gene_names, gnn_idx_map, True)
        self._collate = build_collate_fn(tokenizer, all_gene_names)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.micro_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            collate_fn=self._collate, persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
            collate_fn=self._collate, persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
            collate_fn=self._collate, persistent_workers=self.num_workers > 0,
        )


# ─── Model ────────────────────────────────────────────────────────────────────
class TripleStreamFusionHead(nn.Module):
    """Fuse three streams: AIDO.Cell gene-pos + summary + STRING_GNN PPI."""

    def __init__(self, aido_dim: int = 640, gnn_dim: int = 256, out_dim: int = 640):
        super().__init__()
        in_dim = aido_dim * 2 + gnn_dim  # 640 + 640 + 256 = 1536
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        self.act = nn.GELU()
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, gene_emb, summary_emb, gnn_feat):
        combined = torch.cat([gene_emb, summary_emb, gnn_feat], dim=-1)  # [B, 1536]
        return self.act(self.proj(self.norm(combined)))  # [B, 640]


class TripleStreamHybridModel(nn.Module):
    """
    Triple-stream perturbation response predictor:
      - AIDO.Cell-100M backbone (LoRA) for transcriptomic context
      - STRING_GNN frozen PPI embeddings for network topology context
      - Bilinear output head for 6,640-gene ternary classification
    """

    def __init__(
        self,
        gnn_emb_table: torch.Tensor,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        bilinear_rank: int = 256,
        head_dropout: float = 0.15,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        hidden_dim: int = AIDO_HIDDEN_DIM,
    ):
        super().__init__()

        # ── AIDO.Cell-100M backbone with LoRA ──────────────────────────────
        backbone = AutoModel.from_pretrained(
            AIDO_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # ── STRING_GNN embeddings: frozen buffer [N_nodes, 256] ────────────
        # Precomputed once in setup(); excluded from checkpoint (recomputed on load)
        self.register_buffer("gnn_emb_table", gnn_emb_table)

        # Learnable OOV embedding for perturbed genes absent from STRING_GNN
        self.oov_gnn_emb = nn.Parameter(torch.zeros(GNN_DIM))
        nn.init.normal_(self.oov_gnn_emb, std=0.01)

        # ── Triple-stream fusion: [1536] → [640] ───────────────────────────
        self.fusion = TripleStreamFusionHead(
            aido_dim=hidden_dim, gnn_dim=GNN_DIM, out_dim=hidden_dim
        )

        # ── Bilinear prediction head ────────────────────────────────────────
        self.dropout = nn.Dropout(head_dropout)
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * bilinear_rank, bias=True)
        self.out_gene_emb = nn.Embedding(n_genes_out, bilinear_rank)
        nn.init.normal_(self.out_gene_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)

        self.n_classes = n_classes
        self.bilinear_rank = bilinear_rank
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask, gene_pos, gnn_idx):
        # ── AIDO.Cell backbone ─────────────────────────────────────────────
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, 19266, 640] in bfloat16
        B = input_ids.shape[0]
        device = hidden.device

        # Stream A: gene-specific position embedding
        valid_aido = gene_pos >= 0
        safe_pos = gene_pos.clone()
        safe_pos[~valid_aido] = 0
        gene_emb = hidden[torch.arange(B, device=device), safe_pos]
        if (~valid_aido).any():
            mean_emb = hidden[:, :AIDO_N_GENES, :].mean(dim=1)
            gene_emb = gene_emb.clone()
            gene_emb[~valid_aido] = mean_emb[~valid_aido]

        # Stream B: summary token mean (last 2 positions appended by AIDO.Cell)
        summary_emb = hidden[:, AIDO_N_GENES:, :].mean(dim=1)

        # Cast AIDO.Cell outputs to float32 for stable downstream computation
        gene_emb = gene_emb.float()
        summary_emb = summary_emb.float()

        # Stream C: STRING_GNN PPI embedding (frozen precomputed buffer)
        valid_gnn = gnn_idx >= 0
        safe_gnn = gnn_idx.clamp(min=0)
        gnn_feat = self.gnn_emb_table[safe_gnn].float()  # [B, 256]
        # Use torch.where to handle OOV genes — avoids CUDA in-place boolean-mask
        # assignment issues when value shape [256] != indexed region shape [k, 256].
        oov_emb_expanded = self.oov_gnn_emb.float().unsqueeze(0).expand_as(gnn_feat)  # [B, 256]
        oov_mask_expanded = (~valid_gnn).unsqueeze(-1).expand_as(gnn_feat)            # [B, 256]
        gnn_feat = torch.where(oov_mask_expanded, oov_emb_expanded, gnn_feat)

        # ── Fusion: [B, 1536] → [B, 640] ──────────────────────────────────
        fused = self.fusion(gene_emb, summary_emb, gnn_feat)
        fused = self.dropout(fused)

        # ── Bilinear head: [B, 640] → [B, n_classes, n_genes_out] ──────────
        proj = self.proj_bilinear(fused).view(B, self.n_classes, self.bilinear_rank)
        logits = torch.einsum("bcr,gr->bcg", proj, self.out_gene_emb.weight)

        return logits  # [B, 3, 6640]


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
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        bilinear_rank: int = 256,
        head_dropout: float = 0.15,
        lr_backbone: float = 3e-5,
        lr_head: float = 5e-4,
        weight_decay: float = 1e-3,
        warmup_steps: int = 100,
        focal_gamma: float = 2.0,
        max_steps_total: int = 10000,
        label_smoothing: float = 0.05,
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

        # Precompute STRING_GNN PPI embeddings once per process (~18.5 MB, fast)
        self.print("Precomputing STRING_GNN PPI embeddings...")
        gnn_emb, _ = precompute_string_gnn_embeddings()
        self.print(f"STRING_GNN embeddings shape: {gnn_emb.shape}")

        self.model = TripleStreamHybridModel(
            gnn_emb_table=gnn_emb,
            lora_r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            bilinear_rank=hp.bilinear_rank,
            head_dropout=hp.head_dropout,
        )

        # Cast trainable parameters to float32 for stable optimization
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        self.register_buffer("class_weights", CLASS_WEIGHTS)

    def forward(self, input_ids, attention_mask, gene_pos, gnn_idx):
        return self.model(input_ids, attention_mask, gene_pos, gnn_idx)

    def _compute_loss(self, logits, labels):
        return focal_cross_entropy(
            logits, labels,
            class_weights=self.class_weights,
            gamma=self.hparams.focal_gamma,
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_pos"], batch["gnn_idx"],
        )
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_pos"], batch["gnn_idx"],
        )
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
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_pos"], batch["gnn_idx"],
        )
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
        backbone_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and "backbone" in n
        ]
        head_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and "backbone" not in n
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": hp.lr_backbone},
                {"params": head_params, "lr": hp.lr_head},
            ],
            weight_decay=hp.weight_decay,
        )

        def lr_lambda(current_step):
            if current_step < hp.warmup_steps:
                return float(current_step) / max(1, hp.warmup_steps)
            progress = float(current_step - hp.warmup_steps) / max(
                1, hp.max_steps_total - hp.warmup_steps
            )
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save trainable params + small buffers; exclude large precomputed gnn_emb_table."""
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        # Include class_weights buffer but exclude the large gnn_emb_table (~18.5 MB)
        buffer_keys = {
            prefix + n for n, _ in self.named_buffers()
            if "gnn_emb_table" not in n
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
        """Load trainable params; gnn_emb_table is recomputed in setup()."""
        return super().load_state_dict(state_dict, strict=False)


# ─── Argument Parsing ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="node1-1-3-1-1: Triple-stream AIDO.Cell-100M + STRING_GNN hybrid"
    )
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--bilinear-rank", type=int, default=256)
    p.add_argument("--head-dropout", type=float, default=0.15)
    p.add_argument("--lr-backbone", type=float, default=3e-5)
    p.add_argument("--lr-head", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=25)
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

    # Compute cosine schedule total steps aligned to actual training length
    _train_df_size = pd.read_csv(
        Path(args.data_dir) / "train.tsv", sep="\t", usecols=["pert_id"]
    ).shape[0]
    steps_per_epoch = _train_df_size // (args.micro_batch_size * n_gpus)
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)
    max_steps_total = effective_steps_per_epoch * args.max_epochs

    lit = PerturbationLitModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bilinear_rank=args.bilinear_rank,
        head_dropout=args.head_dropout,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        focal_gamma=args.focal_gamma,
        max_steps_total=max(max_steps_total, 1),
        label_smoothing=args.label_smoothing,
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
