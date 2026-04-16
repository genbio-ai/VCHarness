"""
Node 2-1-2-1-1 — ESM2 Protein Language Model + Bilinear Head
               (Breaking STRING_GNN Ceiling via Sequence-based Gene Representations)

Architecture:
  - ESM2-650M (frozen, precomputed once per setup) → protein sequence embeddings [1280-dim]
  - Protein sequences retrieved from GENCODE hg38 protein FASTA by ENSG ID
  - OOV embedding (learnable) for genes without protein sequences
  - 6-layer residual MLP head (hidden=512, expand=4, rank=256, dropout=0.2)
  - Bilinear interaction: pert_repr [B, 3, 256] × out_gene_emb [6640, 256]
  - Class-weighted focal loss (gamma=2.0, weights=[down=1.5, neutral=0.8, up=2.5])
  - AdamW: lr=5e-4, wd=1e-3, cosine annealing (warmup=100, total=6600 steps)
  - Patience=50, gradient clipping=1.0

Key Design Rationale:
  - Parent (node2-1-2-1): F1=0.5016 — STRING_GNN lineage plateau at ~0.501
  - STRING_GNN PPI topology has been exhausted across 5+ nodes without breaking F1=0.502
  - PRIMARY recommendation from feedback: protein language model backbone
  - ESM2 captures complementary sequence-level biology:
    * Protein domains, conserved motifs, evolutionary context
    * Perturbation sensitivity often tied to protein function, not just network topology
  - Frozen precomputed approach (same as STRING_GNN's success in node2-1-2):
    * No GPU overhead during training (ESM2 model discarded after precomputation)
    * Disk cache for fast re-runs: run/esm2_emb_cache.pt
  - Class weights reverted to node2-1-2's proven [1.5, 0.8, 2.5] (not aggressive [2.0, 0.5, 4.0])
  - Warmup increased from 50 to 100 steps as recommended by feedback
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
from lightning.pytorch.strategies import DDPStrategy
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EsmForMaskedLM

# ─── Constants ────────────────────────────────────────────────────────────────

ESM2_MODEL_NAME     = "facebook/esm2_t33_650M_UR50D"
ESM2_DIM            = 1280        # 650M hidden size
GENCODE_PROTEIN_FA  = "/home/data/genome/hg38_gencode_protein.fa"

N_GENES_OUT   = 6640
N_CLASSES     = 3
HEAD_HIDDEN   = 512
HEAD_EXPAND   = 4
BILINEAR_RANK = 256


# ─── FASTA utilities ──────────────────────────────────────────────────────────

def _iter_fasta(filepath: str):
    """Lightweight FASTA parser — yields (header_without_gt, full_sequence) tuples."""
    with open(filepath) as fh:
        header: Optional[str] = None
        seq_parts: List[str] = []
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            yield header, "".join(seq_parts)


def build_ensg_to_protein_seq(gencode_protein_fa: str) -> Dict[str, str]:
    """
    Build {ensg_base_id: longest_protein_seq} from GENCODE protein FASTA.

    GENCODE header format:
    >ENSP00000493376.2|ENST00000641515.2|ENSG00000186092.7|...|OR4F5|326
     ^index 0           ^index 1           ^index 2

    We strip the version suffix from the ENSG ID (e.g. ENSG00000186092.7 → ENSG00000186092)
    to match the dataset's pert_id column (which has no version).
    For genes with multiple protein isoforms, we keep the longest sequence.
    """
    ensg_to_seq: Dict[str, str] = {}
    for header, seq in _iter_fasta(gencode_protein_fa):
        parts = header.split("|")
        if len(parts) < 3:
            continue
        ensg_base = parts[2].split(".")[0]          # e.g. "ENSG00000186092"
        if ensg_base not in ensg_to_seq or len(seq) > len(ensg_to_seq[ensg_base]):
            ensg_to_seq[ensg_base] = seq
    return ensg_to_seq


# ─── ESM2 Embedding Precomputation ────────────────────────────────────────────

def precompute_esm2_embeddings(
    pert_ids: List[str],
    ensg_to_seq: Dict[str, str],
    device: torch.device,
    model_name: str = ESM2_MODEL_NAME,
    esm2_dim: int = ESM2_DIM,
    max_len: int = 1024,
    batch_size: int = 16,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Precompute frozen ESM2 protein embeddings for all unique pert_ids.

    Embedding = mean-pooled last-layer hidden states (excluding special tokens).
    Genes without protein sequences are not included (caller uses learnable oov_emb).

    Returns:
        emb_cache : float32 CPU tensor [N_with_seq, esm2_dim]
        id_to_idx : {pert_id: row_index_in_emb_cache} for genes that have sequences
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Rank-0-first download pattern (required by code spec)
    if local_rank == 0:
        AutoTokenizer.from_pretrained(model_name)
        EsmForMaskedLM.from_pretrained(model_name, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    esm_model  = EsmForMaskedLM.from_pretrained(model_name, dtype=torch.float32)
    esm_model  = esm_model.eval().to(device)

    # Gather unique pert_ids preserving order
    unique_ids     = list(dict.fromkeys(pert_ids))
    seqs_with_ids  = [(pid, ensg_to_seq[pid]) for pid in unique_ids if pid in ensg_to_seq]
    n_total        = len(unique_ids)
    n_with_seq     = len(seqs_with_ids)

    print(f"[ESM2 rank {local_rank}] {n_with_seq}/{n_total} genes have protein sequences "
          f"({n_total - n_with_seq} OOV → learnable embedding)")

    all_embeddings: List[torch.Tensor] = []
    id_to_idx:      Dict[str, int]     = {}

    for b_start in range(0, n_with_seq, batch_size):
        batch_items = seqs_with_ids[b_start : b_start + batch_size]
        batch_ids   = [x[0] for x in batch_items]
        batch_seqs  = [x[1] for x in batch_items]

        tokens = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,    # includes <cls> and <eos>
        )
        input_ids      = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            output = esm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Last-layer hidden states: [B, seq_len, esm2_dim]
        hidden = output["hidden_states"][-1].float()

        # Mask for non-special tokens (exclude <cls>=0, <eos>=2, <pad>=1)
        special_ids = torch.tensor(
            [tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id],
            device=device,
        )
        non_special = ~torch.isin(input_ids, special_ids)          # [B, seq_len]
        masked_h    = hidden * non_special.unsqueeze(-1).float()    # [B, seq_len, D]
        counts      = non_special.float().sum(dim=1, keepdim=True).clamp(min=1e-9)  # [B,1]
        mean_emb    = (masked_h.sum(dim=1) / counts).cpu()         # [B, esm2_dim]

        for i, pid in enumerate(batch_ids):
            id_to_idx[pid] = len(all_embeddings)
            all_embeddings.append(mean_emb[i])

        done = min(b_start + batch_size, n_with_seq)
        if (b_start // batch_size + 1) % 20 == 0 or done == n_with_seq:
            print(f"[ESM2 rank {local_rank}] Progress: {done}/{n_with_seq}")

    if all_embeddings:
        emb_cache = torch.stack(all_embeddings, dim=0).float()      # [N_with_seq, dim]
    else:
        emb_cache = torch.zeros(0, esm2_dim, dtype=torch.float32)

    # Free ESM2 from GPU memory
    esm_model.cpu()
    del esm_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[ESM2 rank {local_rank}] Cache ready: {emb_cache.shape}")
    return emb_cache, id_to_idx


# ─── Focal Loss with Class Weights ────────────────────────────────────────────

class FocalLossWithWeights(nn.Module):
    """
    Focal loss with optional per-class weights.

    class_weights = [down=1.5, neutral=0.8, up=2.5] — proven in node2-1-2 (F1=0.5011).
    gamma=2.0 — standard focal modulation proven across the tree.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: [N, C]; targets: [N] long"""
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        with torch.no_grad():
            pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


# ─── Metric (matches calc_metric.py) ──────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Per-gene macro F1 matching calc_metric.py.  pred_np: [N,3,G], labels_np: [N,G]."""
    pred_cls = pred_np.argmax(axis=1)
    f1_vals  = []
    for g in range(labels_np.shape[1]):
        yt      = labels_np[:, g]
        yh      = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1   = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Bilinear Head ────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """LN → Linear(D→D*expand) → GELU → Dropout → Linear(D*expand→D) + skip."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ESM2BilinearHead(nn.Module):
    """
    Bilinear prediction head adapted for ESM2 input embeddings (1280-dim).

    Architecture:
      input [B, 1280]
        → Linear(1280→512)          [proj_in]
        → 6 × ResidualBlock(512, expand=4, dropout=0.2)
        → Linear(512→3*256=768)     [proj_out]
        → reshape [B, 3, 256]
        → einsum("bcr,gr->bcg", [B,3,256], out_gene_emb[6640,256])
        → logits [B, 3, 6640]

    rank=256: same as node2-1-2 (F1=0.5011, proven good baseline).
    rank=512 (tried in node2-1-2-1) did not improve over rank=256.
    """

    def __init__(
        self,
        in_dim:    int = ESM2_DIM,       # 1280
        hidden:    int = HEAD_HIDDEN,    # 512
        expand:    int = HEAD_EXPAND,    # 4
        n_blocks:  int = 6,
        dropout:   float = 0.2,
        rank:      int = BILINEAR_RANK,  # 256
        n_genes:   int = N_GENES_OUT,    # 6640
        n_classes: int = N_CLASSES,      # 3
    ):
        super().__init__()
        self.rank      = rank
        self.n_classes = n_classes
        self.n_genes   = n_genes

        self.proj_in  = nn.Linear(in_dim, hidden)
        self.blocks   = nn.ModuleList([
            ResidualBlock(hidden, expand, dropout) for _ in range(n_blocks)
        ])
        self.proj_out     = nn.Linear(hidden, n_classes * rank)
        # Learnable output gene embeddings [6640, 256] — each gene gets its own vector
        self.out_gene_emb = nn.Parameter(torch.randn(n_genes, rank) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_dim]  →  logits: [B, 3, 6640]"""
        h    = self.proj_in(x)              # [B, 512]
        for block in self.blocks:
            h = block(h)                    # [B, 512]
        proj  = self.proj_out(h)            # [B, 3*256=768]
        B     = proj.shape[0]
        pert_proj = proj.view(B, self.n_classes, self.rank)   # [B, 3, 256]
        logits    = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)  # [B,3,6640]
        return logits


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbDataset(Dataset):
    """Simple dataset — just pert_ids, symbols, and optional labels."""

    def __init__(
        self,
        pert_ids: List[str],
        symbols:  List[str],
        labels:   Optional[torch.Tensor] = None,  # [N, 6640] long {0,1,2}
    ):
        self.pert_ids = pert_ids
        self.symbols  = symbols
        self.labels   = labels

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {"pert_id": self.pert_ids[idx], "symbol": self.symbols[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    out = {
        "pert_id": [b["pert_id"] for b in batch],
        "symbol":  [b["symbol"]  for b in batch],
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir:         str = "data",
        micro_batch_size: int = 16,
        num_workers:      int = 4,
    ):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage: Optional[str] = None):
        def load_split(fname: str, has_label: bool) -> PerturbDataset:
            df       = pd.read_csv(self.data_dir / fname, sep="\t")
            pert_ids = df["pert_id"].tolist()
            symbols  = df["symbol"].tolist()
            labels   = None
            if has_label and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return PerturbDataset(pert_ids, symbols, labels)

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  False)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── DDP gather utility ───────────────────────────────────────────────────────

def _gather_tensors(
    local_p: torch.Tensor,
    local_l: torch.Tensor,
    device:  torch.device,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))

    pad = max_sz - local_p.shape[0]
    p   = local_p.to(device)
    l   = local_l.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], 0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], 0)

    gp = [torch.zeros_like(p) for _ in range(world_size)]
    gl = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(gp, p)
    dist.all_gather(gl, l)

    rp = torch.cat([gp[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    rl = torch.cat([gl[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    return rp, rl


# ─── LightningModule ──────────────────────────────────────────────────────────

class ESM2LitModule(pl.LightningModule):

    def __init__(
        self,
        lr:                   float = 5e-4,
        weight_decay:         float = 1e-3,
        focal_gamma:          float = 2.0,
        class_weight_down:    float = 1.5,
        class_weight_neutral: float = 0.8,
        class_weight_up:      float = 2.5,
        head_dropout:         float = 0.2,
        warmup_steps:         int   = 100,   # increased from 50 (feedback recommendation)
        total_steps:          int   = 6600,
        data_dir:             str   = "data",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Accumulator lists for val/test steps
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str]          = []
        self._test_symbols:  List[str]          = []
        self._test_labels:   List[torch.Tensor] = []

        # These are populated in setup()
        self._emb_cache: Optional[torch.Tensor] = None  # [N_genes, ESM2_DIM] on device
        self._id_to_idx: Dict[str, int]          = {}

    # ------------------------------------------------------------------
    #  Setup: ESM2 precomputation + model init
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None):
        data_dir   = Path(self.hparams.data_dir)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        out_dir    = Path(__file__).parent / "run"
        cache_path = out_dir / "esm2_emb_cache.pt"

        # ── 1. Load GENCODE protein sequences ─────────────────────────────────
        print(f"[Setup rank {local_rank}] Loading GENCODE protein sequences from {GENCODE_PROTEIN_FA} ...")
        ensg_to_seq = build_ensg_to_protein_seq(GENCODE_PROTEIN_FA)
        print(f"[Setup rank {local_rank}] Loaded {len(ensg_to_seq)} protein sequences")

        # ── 2. Collect all unique pert_ids (train + val + test) ───────────────
        all_pert_ids: List[str] = []
        for fname in ["train.tsv", "val.tsv", "test.tsv"]:
            fpath = data_dir / fname
            if fpath.exists():
                df = pd.read_csv(fpath, sep="\t")
                all_pert_ids.extend(df["pert_id"].tolist())

        # ── 3. Load or compute ESM2 embeddings ────────────────────────────────
        if cache_path.exists():
            print(f"[Setup rank {local_rank}] Loading ESM2 cache from disk: {cache_path}")
            saved     = torch.load(cache_path, map_location="cpu")
            emb_cache = saved["emb_cache"]
            id_to_idx = saved["id_to_idx"]
        else:
            print(f"[Setup rank {local_rank}] No cache found. Computing ESM2 embeddings...")
            emb_cache, id_to_idx = precompute_esm2_embeddings(
                pert_ids=all_pert_ids,
                ensg_to_seq=ensg_to_seq,
                device=self.device,
                model_name=ESM2_MODEL_NAME,
                esm2_dim=ESM2_DIM,
                batch_size=16,
            )
            # Rank-0 saves the cache
            if local_rank == 0:
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"emb_cache": emb_cache, "id_to_idx": id_to_idx}, cache_path)
                print(f"[Setup rank 0] Saved ESM2 cache → {cache_path}")
            if dist.is_available() and dist.is_initialized():
                dist.barrier()   # Wait for cache to be written before other runs might read it

        # Move cache to this rank's device
        self._emb_cache = emb_cache.to(self.device)
        self._id_to_idx = id_to_idx

        # ── 4. Learnable OOV embedding for genes without protein sequences ─────
        self.oov_emb = nn.Parameter(
            torch.randn(1, ESM2_DIM, device=self.device) * 0.01
        )

        # ── 5. Bilinear prediction head ───────────────────────────────────────
        self.head = ESM2BilinearHead(
            in_dim=ESM2_DIM,
            hidden=HEAD_HIDDEN,
            expand=HEAD_EXPAND,
            n_blocks=6,
            dropout=self.hparams.head_dropout,
            rank=BILINEAR_RANK,
            n_genes=N_GENES_OUT,
            n_classes=N_CLASSES,
        ).to(self.device)

        # ── 6. Focal loss ─────────────────────────────────────────────────────
        cw = torch.tensor(
            [self.hparams.class_weight_down,
             self.hparams.class_weight_neutral,
             self.hparams.class_weight_up],
            dtype=torch.float32,
        )
        self.focal_loss = FocalLossWithWeights(
            gamma=self.hparams.focal_gamma,
            class_weights=cw,
        ).to(self.device)

        # ── 7. Cast trainable parameters to float32 ───────────────────────────
        for p in self.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        total_params = sum(p.numel() for p in self.parameters())
        trainable    = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Setup rank {local_rank}] Model ready: {trainable:,}/{total_params:,} trainable params")

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def _get_pert_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Look up frozen ESM2 embeddings; use oov_emb for missing genes."""
        indices   = [self._id_to_idx.get(pid, -1) for pid in pert_ids]
        indices_t = torch.tensor(indices, device=self.device)
        oov_mask  = (indices_t < 0)

        safe_idx = indices_t.clone()
        safe_idx[oov_mask] = 0   # temporary valid index

        if self._emb_cache is not None and self._emb_cache.shape[0] > 0:
            embs = self._emb_cache[safe_idx].float()    # [B, ESM2_DIM]
        else:
            embs = self.oov_emb.expand(len(pert_ids), -1).float()

        if oov_mask.any():
            embs = embs.clone()
            embs[oov_mask] = self.oov_emb.expand(oov_mask.sum(), -1)

        return embs    # [B, ESM2_DIM]

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        emb    = self._get_pert_emb(pert_ids)   # [B, ESM2_DIM]
        logits = self.head(emb)                  # [B, 3, 6640]
        return logits

    # ------------------------------------------------------------------
    #  Loss helper
    # ------------------------------------------------------------------

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, 3, 6640] → [B*6640, 3];  labels: [B, 6640] → [B*6640]
        logits_2d = logits.float().permute(0, 2, 1).reshape(-1, N_CLASSES)
        labels_1d = labels.reshape(-1)
        return self.focal_loss(logits_2d, labels_1d)

    # ------------------------------------------------------------------
    #  Train / Val / Test steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        logits = self(batch["pert_id"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["pert_id"])
        if "label" in batch:
            loss = self._loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["pert_id"])
        probs  = torch.softmax(logits.float(), dim=1)
        self._test_preds.append(probs.detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs  = torch.cat(self._test_preds, 0)
        dummy_labels = (
            torch.cat(self._test_labels, 0) if self._test_labels
            else torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        )

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
            all_pert, all_syms    = self._test_pert_ids, self._test_symbols

        if self.trainer.is_global_zero:
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate by pert_id (DDP padding may introduce duplicates)
            seen_pids:    set       = set()
            dedup_indices: List[int] = []
            for i, pid in enumerate(all_pert):
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    dedup_indices.append(i)

            all_probs_np = all_probs.numpy()
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for i in dedup_indices:
                    fh.write(
                        f"{all_pert[i]}\t{all_syms[i]}\t"
                        f"{json.dumps(all_probs_np[i].tolist())}\n"
                    )
            self.print(f"[Node2-1-2-1-1] Saved {len(dedup_indices)} predictions → {pred_path}")

            if self._test_labels:
                dedup_probs  = all_probs_np[dedup_indices]
                dedup_labels = all_labels[dedup_indices].numpy()
                f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
                self.print(f"[Node2-1-2-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    # ------------------------------------------------------------------
    #  Optimizer / Scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        hp = self.hparams
        # All trainable params: head (out_gene_emb, residual blocks, proj_in/out) + oov_emb
        optimizer = torch.optim.AdamW(
            list(self.head.parameters()) + [self.oov_emb],
            lr=hp.lr,
            weight_decay=hp.weight_decay,
            betas=(0.9, 0.999),
        )

        warmup = hp.warmup_steps
        total  = hp.total_steps

        def lr_lambda(current_step: int):
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            progress = float(current_step - warmup) / float(max(1, total - warmup))
            return max(1e-7, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # ------------------------------------------------------------------
    #  Checkpoint management — save only trainable params + buffers
    # ------------------------------------------------------------------

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters (head + oov_emb) and focal_loss buffers.
        The ESM2 embedding cache is excluded (recomputed deterministically in setup)."""
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        sd = {k: v for k, v in full_sd.items()
              if k in trainable_keys or k in buffer_keys}

        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving ckpt: {trained:,}/{total:,} trainable params ({100*trained/total:.2f}%)"
        )
        return sd

    def load_state_dict(self, state_dict, strict=True):
        """Load partial checkpoint (trainable params only); strict=False for compatibility."""
        return super().load_state_dict(state_dict, strict=False)


# ─── Argument Parser ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 2-1-2-1-1 — ESM2 Protein Language Model + Bilinear Head"
    )
    p.add_argument("--data-dir",             type=str,   default="data")
    p.add_argument("--lr",                   type=float, default=5e-4,
                   help="AdamW learning rate for head parameters")
    p.add_argument("--weight-decay",         type=float, default=1e-3)
    p.add_argument("--focal-gamma",          type=float, default=2.0)
    p.add_argument("--class-weight-down",    type=float, default=1.5,
                   help="Focal class weight for down-regulated (8.1%)")
    p.add_argument("--class-weight-neutral", type=float, default=0.8,
                   help="Focal class weight for neutral (88.9%)")
    p.add_argument("--class-weight-up",      type=float, default=2.5,
                   help="Focal class weight for up-regulated (3.0%)")
    p.add_argument("--head-dropout",         type=float, default=0.2)
    p.add_argument("--warmup-steps",         type=int,   default=100,
                   help="LR warmup steps (increased from 50 per feedback)")
    p.add_argument("--total-steps",          type=int,   default=6600,
                   help="Total steps for cosine LR annealing")
    p.add_argument("--micro-batch-size",     type=int,   default=16,
                   help="Micro batch size per GPU")
    p.add_argument("--global-batch-size",    type=int,   default=128,
                   help="Global batch size (must be multiple of micro_batch_size * 8)")
    p.add_argument("--max-epochs",           type=int,   default=200)
    p.add_argument("--patience",             type=int,   default=50,
                   help="EarlyStopping patience")
    p.add_argument("--num-workers",          type=int,   default=4)
    p.add_argument("--val-check-interval",   type=float, default=1.0)
    p.add_argument("--debug-max-step",       type=int,   default=None,
                   help="Limit train/val/test steps for debug mode")
    p.add_argument("--fast-dev-run",         action="store_true", default=False)
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm = PerturbDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    lit = ESM2LitModule(
        lr=args.lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        class_weight_down=args.class_weight_down,
        class_weight_neutral=args.class_weight_neutral,
        class_weight_up=args.class_weight_up,
        head_dropout=args.head_dropout,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        data_dir=args.data_dir,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-5)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    # Debug / fast_dev_run configuration
    max_steps_trainer: int        = -1
    limit_train: float | int      = 1.0
    limit_val:   float | int      = 1.0
    limit_test:  float | int      = 1.0
    fast_dev_run                  = False

    if args.debug_max_step is not None:
        max_steps_trainer = args.debug_max_step
        limit_train = args.debug_max_step
        limit_val   = 2
        limit_test  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    # find_unused_parameters=True: oov_emb may not receive gradients when all
    # genes have protein sequences; DDP must skip unused param synchronization.
    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps_trainer,
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
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    trainer.fit(lit, datamodule=dm)

    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 2-1-2-1-1 — ESM2 Protein Language Model + Bilinear Head\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
