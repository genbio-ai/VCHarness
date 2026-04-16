"""Node: node3-1-1-1-3-1 — Frozen ESM2-650M + Frozen STRING_GNN Dual-Branch + Gated Fusion + Manifold Mixup

Architecture Overview:
  - Frozen ESM2-650M precomputed embeddings (1280-dim): rich protein language model semantics
  - Frozen STRING_GNN precomputed embeddings (256-dim): PPI topology signal
  - Learnable sigmoidal gated fusion (FUSION_DIM=512): weighted combination of both branches
  - 3-block PreNorm Residual MLP (hidden_dim=384): proven optimal capacity for 1273 samples
  - Muon optimizer (fc1/fc2 in blocks) + AdamW for all other params
  - CosineAnnealingWarmRestarts (T_0=80, T_mult=2): proven to reach F1=0.5243 in node3-3-1-2-1-1-1
  - Manifold Mixup (prob=0.65, alpha=0.2): proven data augmentation in 0.52+ F1 nodes
  - Top-3 checkpoint ensemble: proven better than top-7 ensemble (node3-1-1-1-1-2-1-1-1 F1=0.5283)
  - Weighted cross-entropy + label smoothing (NO focal loss — incompatible with Muon)
  - Gradient clipping max_norm=1.0
  - NO per-gene bias (removed: identified as memorization vector in parent feedback)

Key Improvements vs Parent (node3-1-1-1-3, F1=0.429):
  1. ARCHITECTURE: ESM2-650M + STRING_GNN dual-branch (proven to reach 0.52+ vs STRING-only 0.48 ceiling)
  2. FEATURES: ESM2-650M precomputed embeddings (protein sequence semantics)
  3. FUSION: Gated sigmoidal fusion (proven: node3-3-1-2-1-1-1 F1=0.5243, node3-1-1-1-1-2-1-1 F1=0.5265)
  4. SCHEDULE: CosineAnnealingWarmRestarts T_0=80 (vs RLROP — warm restarts escape local minima)
  5. AUGMENT: Manifold Mixup prob=0.65 (improves generalization on 1273 training samples)
  6. ENSEMBLE: Top-3 threshold ensemble (node3-1-1-1-1-2-1-1-1 proved F1=0.5283 > 0.5265 top-7)
  7. REGULARIZATION: Remove per-gene bias (20K memorization-prone params), weight_decay=1e-3
  8. CORRECT val/f1: all_gather + dedup (maintained from parent)
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import time
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
from transformers import AutoModel, AutoTokenizer, EsmModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRING_GNN_DIR = "/home/Models/STRING_GNN"
ESM2_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
N_GENES = 6640
N_CLASSES = 3
STRING_EMB_DIM = 256    # STRING_GNN output dim
ESM2_EMB_DIM = 1280     # ESM2-650M embedding dim
FUSION_DIM = 512        # Proven optimal in node3-1-1-1-1-2-1-1 (F1=0.5265)


# ---------------------------------------------------------------------------
# PreLN Residual Block
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """PreLN residual block: LayerNorm → Linear → GELU → Dropout → Linear → Dropout + residual."""

    def __init__(self, dim: int, expand: int = 2, dropout: float = 0.30) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expand)
        self.fc2 = nn.Linear(dim * expand, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


# ---------------------------------------------------------------------------
# Dual-Branch Fusion Model
# ---------------------------------------------------------------------------
class DualBranchModel(nn.Module):
    """ESM2-650M + STRING_GNN frozen dual-branch with gated fusion → 3-block PreNorm MLP.

    Proven architecture from node3-3-1-2-1-1-1 (F1=0.5243) and node3-1-1-1-1-2-1-1 (F1=0.5265):
    - ESM2 branch: 1280 → FUSION_DIM projection
    - STRING branch: 256 → FUSION_DIM projection
    - Sigmoidal gate: learned weighting of both branches
    - 3-block PreNorm MLP (hidden_dim=384): optimal capacity for 1273 samples
    - Unfactorized Linear(384 → 19920) output head
    - Head dropout p=0.15: proven regularization from node1-3-2-2-1
    - NO per-gene bias: removed based on parent feedback (20K memorization-prone params)
    """

    def __init__(
        self,
        esm2_dim: int = ESM2_EMB_DIM,
        string_dim: int = STRING_EMB_DIM,
        fusion_dim: int = FUSION_DIM,
        hidden_dim: int = 384,
        n_genes: int = N_GENES,
        n_blocks: int = 3,
        dropout: float = 0.30,
        head_dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes

        # ESM2 projection branch: 1280 → fusion_dim
        self.esm2_proj = nn.Sequential(
            nn.LayerNorm(esm2_dim),
            nn.Linear(esm2_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # STRING_GNN projection branch: 256 → fusion_dim
        self.string_proj = nn.Sequential(
            nn.LayerNorm(string_dim),
            nn.Linear(string_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Sigmoidal gate: learnable weighted combination of both branches
        # gate = sigmoid(W * concat([esm2_feat, string_feat]))
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid(),
        )

        # Input projection from fusion → hidden_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3-block PreNorm Residual MLP trunk
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=2, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Output head with targeted head dropout
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.head_dropout = nn.Dropout(head_dropout)
        self.out_proj = nn.Linear(hidden_dim, n_genes * N_CLASSES)
        # NO per-gene bias: identified as memorization source in parent node3-1-1-1-3 feedback

    def forward(
        self,
        esm2_feats: torch.Tensor,   # [B, esm2_dim]
        string_feats: torch.Tensor, # [B, string_dim]
    ) -> torch.Tensor:
        # Project each branch to fusion_dim
        esm2_out = self.esm2_proj(esm2_feats)     # [B, fusion_dim]
        string_out = self.string_proj(string_feats) # [B, fusion_dim]

        # Gated sigmoidal fusion
        concat = torch.cat([esm2_out, string_out], dim=-1)  # [B, 2*fusion_dim]
        gate = self.gate(concat)                              # [B, fusion_dim]
        fused = gate * esm2_out + (1.0 - gate) * string_out # [B, fusion_dim]

        # MLP trunk
        x = self.input_proj(fused)    # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)

        # Output projection
        x = self.output_norm(x)
        x = self.head_dropout(x)
        out = self.out_proj(x)                    # [B, N_GENES * 3]
        return out.view(-1, N_CLASSES, self.n_genes)  # [B, 3, N_GENES]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Maps each perturbed gene to precomputed ESM2 + STRING_GNN feature vectors."""

    def __init__(
        self,
        df: pd.DataFrame,
        esm2_features: torch.Tensor,      # [N_genes_unique, ESM2_EMB_DIM]
        string_features: torch.Tensor,    # [N_string_nodes, STRING_EMB_DIM]
        symbol_to_esm2_idx: Dict[str, int],
        ensg_to_string_idx: Dict[str, int],
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.esm2_features = esm2_features
        self.string_features = string_features
        self.symbol_to_esm2_idx = symbol_to_esm2_idx
        self.ensg_to_string_idx = ensg_to_string_idx

        if "label" in df.columns:
            labels = np.array(
                [json.loads(x) for x in df["label"].tolist()], dtype=np.int64
            )
            self.labels: Optional[torch.Tensor] = torch.tensor(
                labels + 1, dtype=torch.long
            )  # {-1, 0, 1} → {0, 1, 2}
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pert_id = self.pert_ids[idx]
        symbol = self.symbols[idx]

        # ESM2 feature lookup
        esm2_idx = self.symbol_to_esm2_idx.get(symbol, -1)
        if esm2_idx >= 0:
            esm2_feat = self.esm2_features[esm2_idx]
        else:
            esm2_feat = torch.zeros(self.esm2_features.shape[1])

        # STRING_GNN feature lookup
        string_idx = self.ensg_to_string_idx.get(pert_id, -1)
        if string_idx >= 0:
            string_feat = self.string_features[string_idx]
        else:
            string_feat = torch.zeros(self.string_features.shape[1])

        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": pert_id,
            "symbol": symbol,
            "esm2_features": esm2_feat,
            "string_features": string_feat,
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
        micro_batch_size: int = 64,
        num_workers: int = 4,
        esm2_cache_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        # Cache path for precomputed ESM2 embeddings (avoids recomputing on every run)
        self.esm2_cache_path = esm2_cache_path

        self.esm2_features: Optional[torch.Tensor] = None
        self.symbol_to_esm2_idx: Optional[Dict[str, int]] = None
        self.string_features: Optional[torch.Tensor] = None
        self.ensg_to_string_idx: Optional[Dict[str, int]] = None
        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: str = "fit") -> None:
        if self.esm2_features is None:
            self._precompute_all_features()

        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbDataset(
            train_df, self.esm2_features, self.string_features,
            self.symbol_to_esm2_idx, self.ensg_to_string_idx,
        )
        self.val_ds = PerturbDataset(
            val_df, self.esm2_features, self.string_features,
            self.symbol_to_esm2_idx, self.ensg_to_string_idx,
        )
        self.test_ds = PerturbDataset(
            test_df, self.esm2_features, self.string_features,
            self.symbol_to_esm2_idx, self.ensg_to_string_idx,
        )

    def _fetch_uniprot_sequences(self, symbols: List[str]) -> Dict[str, str]:
        """Fetch canonical human protein sequences from UniProt REST API.

        Queries reviewed (Swiss-Prot) human entries by HGNC gene symbol.
        Returns a dict {symbol: sequence}. Missing symbols get empty string.
        """
        import requests

        print(f"Fetching UniProt sequences for {len(symbols)} gene symbols...", flush=True)
        seq_dict: Dict[str, str] = {}
        batch_size = 50
        failed = 0

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            # Build OR query for batch
            gene_query = " OR ".join([f"gene_exact:{s}" for s in batch])
            url = (
                "https://rest.uniprot.org/uniprotkb/search?"
                f"query=({gene_query}) AND organism_id:9606 AND reviewed:true"
                "&fields=gene_names,sequence&format=json&size=500"
            )
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    for entry in data.get("results", []):
                        # Extract gene name (primary name)
                        gene_names = entry.get("genes", [])
                        primary_name = None
                        for gene_info in gene_names:
                            if "geneName" in gene_info:
                                primary_name = gene_info["geneName"]["value"].upper()
                                break
                        if primary_name:
                            seq = entry.get("sequence", {}).get("value", "")
                            if seq and primary_name not in seq_dict:
                                seq_dict[primary_name] = seq
                else:
                    failed += len(batch)
            except Exception as e:
                print(f"Warning: UniProt batch {i//batch_size} failed: {e}", flush=True)
                failed += len(batch)

            # Rate limit: brief pause between batches
            if i + batch_size < len(symbols):
                time.sleep(0.2)

        # Second pass: try individual queries for symbols not found (case variants)
        missing = [s for s in symbols if s.upper() not in seq_dict]
        if missing:
            print(f"Retrying {len(missing)} missing symbols individually...", flush=True)
            for sym in missing:
                try:
                    url = (
                        "https://rest.uniprot.org/uniprotkb/search?"
                        f"query=gene_exact:{sym} AND organism_id:9606 AND reviewed:true"
                        "&fields=gene_names,sequence&format=json&size=1"
                    )
                    resp = requests.get(url, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        for entry in data.get("results", []):
                            seq = entry.get("sequence", {}).get("value", "")
                            if seq:
                                seq_dict[sym.upper()] = seq
                                break
                    time.sleep(0.1)
                except Exception:
                    pass

        found = sum(1 for s in symbols if s.upper() in seq_dict)
        print(
            f"UniProt: found sequences for {found}/{len(symbols)} symbols "
            f"(coverage: {100*found/len(symbols):.1f}%)",
            flush=True,
        )
        return seq_dict

    def _precompute_esm2_embeddings(
        self,
        seq_dict: Dict[str, str],
        symbols: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Precompute ESM2-650M embeddings for all unique gene symbols.

        Returns:
            emb_tensor: [N_symbols, ESM2_EMB_DIM] float32 CPU tensor
            symbol_to_idx: dict mapping symbol (uppercase) → row index in emb_tensor
        """
        import torch.distributed as dist

        device = torch.device("cuda")
        print("Loading ESM2-650M for precomputing frozen protein embeddings...", flush=True)

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        is_dist = dist.is_available() and dist.is_initialized()

        if local_rank == 0:
            # Download/verify on rank 0 first
            AutoTokenizer.from_pretrained(ESM2_MODEL_ID)
        if is_dist:
            dist.barrier()

        tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_ID)
        esm2 = EsmModel.from_pretrained(ESM2_MODEL_ID, dtype=torch.float32).to(device)
        esm2.eval()

        # Process in batches
        batch_size = 32
        max_len = 1024  # ESM2 max context length (including special tokens)
        all_embeddings = []
        symbol_to_idx: Dict[str, int] = {}
        idx = 0

        ordered_symbols = []  # symbols with found sequences
        ordered_seqs = []
        zero_symbols = []      # symbols with no sequence → zero embedding

        for sym in symbols:
            seq = seq_dict.get(sym.upper(), "")
            if seq:
                ordered_symbols.append(sym.upper())
                # Truncate to max_len - 2 (accounting for CLS + EOS)
                ordered_seqs.append(seq[:max_len - 2])
            else:
                zero_symbols.append(sym.upper())

        # Batch inference
        for i in range(0, len(ordered_seqs), batch_size):
            batch_seqs = ordered_seqs[i:i + batch_size]
            batch_syms = ordered_symbols[i:i + batch_size]

            tokens = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(device)

            with torch.no_grad():
                out = esm2(**tokens)

            hidden = out.last_hidden_state  # [B, seq_len, 1280]

            # Mean pool over non-padding tokens (excluding CLS at pos 0 and EOS at last non-pad pos)
            # attention_mask: 1 for real tokens, 0 for padding
            attn_mask = tokens["attention_mask"]  # [B, seq_len]
            # Exclude CLS (first token) and pad tokens
            # Build mask: exclude index 0 (CLS) from pooling
            pool_mask = attn_mask.clone()
            pool_mask[:, 0] = 0  # exclude CLS
            # Also exclude EOS: find last non-pad position
            lengths = attn_mask.sum(dim=1)  # [B]
            for j in range(len(lengths)):
                eos_pos = (lengths[j] - 1).long().item()
                pool_mask[j, eos_pos] = 0  # exclude EOS

            pool_mask_float = pool_mask.float().unsqueeze(-1)  # [B, seq_len, 1]
            pooled = (hidden * pool_mask_float).sum(dim=1) / pool_mask_float.sum(dim=1).clamp(min=1e-9)
            # pooled: [B, 1280]

            for j, sym in enumerate(batch_syms):
                symbol_to_idx[sym] = idx
                all_embeddings.append(pooled[j].float().cpu())
                idx += 1

        # Zero embeddings for missing symbols
        zero_emb = torch.zeros(ESM2_EMB_DIM, dtype=torch.float32)
        for sym in zero_symbols:
            symbol_to_idx[sym] = idx
            all_embeddings.append(zero_emb.clone())
            idx += 1

        emb_tensor = torch.stack(all_embeddings, dim=0)  # [N, 1280]

        del esm2, tokenizer
        torch.cuda.empty_cache()

        print(
            f"ESM2 embeddings precomputed: {emb_tensor.shape} "
            f"({len(ordered_symbols)} with sequence, {len(zero_symbols)} with zeros)",
            flush=True,
        )
        return emb_tensor, symbol_to_idx

    def _precompute_all_features(self) -> None:
        """Precompute both ESM2 and STRING_GNN frozen embeddings.

        Implements barrier-protected precomputation to avoid redundant work in DDP.
        Caches ESM2 embeddings to disk to avoid re-fetching UniProt sequences.
        """
        import torch.distributed as dist

        is_dist = dist.is_available() and dist.is_initialized()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # --- STRING_GNN precomputation ---
        model_dir = Path(STRING_GNN_DIR)
        node_names: List[str] = json.loads((model_dir / "node_names.json").read_text())
        self.ensg_to_string_idx = {name: i for i, name in enumerate(node_names)}

        if is_dist:
            dist.barrier()

        device = torch.device("cuda")
        print("Loading STRING_GNN for precomputing frozen topology embeddings...", flush=True)
        gnn = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device)
        gnn.eval()

        graph = torch.load(model_dir / "graph_data.pt", map_location=device)
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)

        with torch.no_grad():
            out = gnn(
                edge_index=edge_index,
                edge_weight=edge_weight,
                output_hidden_states=False,
            )
            self.string_features = out.last_hidden_state.float().cpu()  # [18870, 256]

        del gnn, graph, out
        torch.cuda.empty_cache()

        print(
            f"STRING_GNN features: {self.string_features.shape} (frozen PPI topology)",
            flush=True,
        )

        # --- ESM2 precomputation ---
        # Collect all unique symbols across all splits
        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")
        all_symbols = list(set(
            train_df["symbol"].tolist()
            + val_df["symbol"].tolist()
            + test_df["symbol"].tolist()
        ))

        # Check if cached embeddings exist
        cache_path = Path(self.esm2_cache_path) if self.esm2_cache_path else None

        if cache_path and cache_path.exists():
            print(f"Loading cached ESM2 embeddings from {cache_path}", flush=True)
            cache = torch.load(cache_path, map_location="cpu")
            self.esm2_features = cache["embeddings"]
            self.symbol_to_esm2_idx = cache["symbol_to_idx"]
            print(f"Loaded ESM2 cache: {self.esm2_features.shape}", flush=True)
        else:
            # Rank 0 fetches sequences and computes embeddings
            if local_rank == 0:
                seq_dict = self._fetch_uniprot_sequences(all_symbols)
                self.esm2_features, self.symbol_to_esm2_idx = self._precompute_esm2_embeddings(
                    seq_dict, all_symbols
                )
                # Save cache
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"embeddings": self.esm2_features, "symbol_to_idx": self.symbol_to_esm2_idx},
                        cache_path,
                    )
                    print(f"Saved ESM2 embeddings cache → {cache_path}", flush=True)

            if is_dist:
                dist.barrier()

            # All non-zero ranks load from cache (rank 0 already computed above)
            if local_rank != 0 and cache_path and cache_path.exists():
                cache = torch.load(cache_path, map_location="cpu")
                self.esm2_features = cache["embeddings"]
                self.symbol_to_esm2_idx = cache["symbol_to_idx"]
            elif local_rank != 0:
                # Broadcast from rank 0 via dist object passing (fallback: recompute)
                # This path is only reached if cache_path not set — should not happen in practice
                seq_dict = self._fetch_uniprot_sequences(all_symbols)
                self.esm2_features, self.symbol_to_esm2_idx = self._precompute_esm2_embeddings(
                    seq_dict, all_symbols
                )

        if is_dist:
            dist.barrier()

    def _make_loader(self, ds: PerturbDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        esm2_dim: int = ESM2_EMB_DIM,
        string_dim: int = STRING_EMB_DIM,
        fusion_dim: int = FUSION_DIM,
        hidden_dim: int = 384,
        n_genes: int = N_GENES,
        n_blocks: int = 3,
        lr: float = 3e-4,
        muon_lr: float = 0.01,
        weight_decay: float = 1e-3,
        dropout: float = 0.30,
        head_dropout: float = 0.15,
        label_smoothing: float = 0.05,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.65,
        t0: int = 80,
        t_mult: int = 2,
        eta_min: float = 1e-6,
        max_epochs: int = 600,
        ensemble_top_k: int = 3,
        ensemble_threshold: float = 0.003,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model: Optional[DualBranchModel] = None

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_pert_ids: List[str] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        self.model = DualBranchModel(
            esm2_dim=self.hparams.esm2_dim,
            string_dim=self.hparams.string_dim,
            fusion_dim=self.hparams.fusion_dim,
            hidden_dim=self.hparams.hidden_dim,
            n_genes=self.hparams.n_genes,
            n_blocks=self.hparams.n_blocks,
            dropout=self.hparams.dropout,
            head_dropout=self.hparams.head_dropout,
        )

        # NOTE: trainable parameters are implicitly in float32 (or bf16 under AMP).
        # Mixed precision (bf16-mixed) is handled by the Lightning Trainer, so no
        # explicit dtype casting is needed here.

        # Class weights: shifted labels {0:down, 1:neutral, 2:up}
        # Frequencies from DATA_ABSTRACT: down=4.77%, neutral=92.82%, up=2.41%
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = 1.0 / freq
        class_weights = class_weights / class_weights.mean()
        self.register_buffer("class_weights", class_weights)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"ESM2-650M+STRING_GNN Dual-Branch | "
            f"trainable={trainable:,}/{total:,} | "
            f"fusion={self.hparams.fusion_dim}, hidden={self.hparams.hidden_dim}, "
            f"dropout={self.hparams.dropout}, head_dropout={self.hparams.head_dropout}"
        )

    def _manifold_mixup(
        self,
        esm2_feats: torch.Tensor,
        string_feats: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Manifold Mixup in the input feature space.

        Proven augmentation strategy from node3-3-1-2-1-1-1 (F1=0.5243):
        Mix in the input embedding space (before the model) with prob=0.65, alpha=0.2.
        Returns (mixed_esm2, mixed_string, labels_a, labels_b, lam) — always 5-tuple.
        """
        lam = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
        batch_size = esm2_feats.size(0)
        idx = torch.randperm(batch_size, device=esm2_feats.device)

        mixed_esm2 = lam * esm2_feats + (1 - lam) * esm2_feats[idx]
        mixed_string = lam * string_feats + (1 - lam) * string_feats[idx]
        labels_b = labels[idx]

        return mixed_esm2, mixed_string, labels, labels_b, lam

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        labels_b: Optional[torch.Tensor] = None,
        lam: float = 1.0,
    ) -> torch.Tensor:
        """Weighted cross-entropy with label smoothing and optional Mixup blending.

        NO focal loss: node1-1-3-1 proved Muon + focal loss = catastrophic collapse (F1=0.191).
        Muon + WCE proved optimal across all best nodes.
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        loss_a = F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )
        if labels_b is not None and lam < 1.0:
            labels_b_flat = labels_b.reshape(-1)
            loss_b = F.cross_entropy(
                logits_flat,
                labels_b_flat,
                weight=self.class_weights,
                label_smoothing=self.hparams.label_smoothing,
            )
            return lam * loss_a + (1.0 - lam) * loss_b
        return loss_a

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        esm2_feats = batch["esm2_features"].to(self.device).float()
        string_feats = batch["string_features"].to(self.device).float()
        labels = batch["label"].to(self.device)  # [B, N_GENES], move to device for randperm

        # Apply Manifold Mixup with probability mixup_prob
        if self.training and np.random.random() < self.hparams.mixup_prob:
            result = self._manifold_mixup(esm2_feats, string_feats, labels)
            esm2_feats, string_feats, labels_a, labels_b, lam = result
            logits = self.model(esm2_feats, string_feats)
            loss = self._compute_loss(logits, labels_a, labels_b, lam)
        else:
            logits = self.model(esm2_feats, string_feats)
            loss = self._compute_loss(logits, labels)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        esm2_feats = batch["esm2_features"].to(self.device).float()
        string_feats = batch["string_features"].to(self.device).float()
        logits = self.model(esm2_feats, string_feats)
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())
        self._val_pert_ids.extend(batch["pert_id"])

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        import torch.distributed as dist

        preds_local = torch.cat(self._val_preds, dim=0)    # [local_N, 3, N_GENES]
        labels_local = torch.cat(self._val_labels, dim=0)  # [local_N, N_GENES]
        local_pert_ids = list(self._val_pert_ids)

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_pert_ids.clear()

        # Gather from ALL ranks for correct global val/f1 computation.
        # With 8 GPUs and 141 val samples, per-GPU F1 is extremely noisy (~18 samples).
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        all_preds = self.all_gather(preds_local)    # [world_size, local_N, 3, N_GENES]
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
        all_labels = self.all_gather(labels_local)  # [world_size, local_N, N_GENES]
        all_labels = all_labels.view(-1, N_GENES)

        # Gather pert_ids for deduplication (DDP pads last batch across ranks)
        gathered_pert_ids = [local_pert_ids]
        if world_size > 1:
            obj_pert = [None] * world_size
            dist.all_gather_object(obj_pert, local_pert_ids)
            gathered_pert_ids = obj_pert

        all_pert_ids_flat = [p for rank_list in gathered_pert_ids for p in rank_list]
        all_preds_np = all_preds.float().cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()

        # Deduplicate by pert_id
        seen: set = set()
        dedup_preds, dedup_labels = [], []
        for i, pid in enumerate(all_pert_ids_flat):
            if pid not in seen:
                seen.add(pid)
                dedup_preds.append(all_preds_np[i])
                dedup_labels.append(all_labels_np[i])

        if dedup_preds:
            f1 = _compute_per_gene_f1(
                np.stack(dedup_preds, axis=0),
                np.stack(dedup_labels, axis=0),
            )
            self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        esm2_feats = batch["esm2_features"].to(self.device).float()
        string_feats = batch["string_features"].to(self.device).float()
        logits = self.model(esm2_feats, string_feats)
        self._test_preds.append(logits.detach().cpu().float())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist

        preds_local = torch.cat(self._test_preds, dim=0)
        labels_local = (
            torch.cat(self._test_labels, dim=0) if self._test_labels else None
        )
        self._test_preds.clear()
        self._test_labels.clear()

        all_preds = self.all_gather(preds_local)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)

        if labels_local is not None:
            all_labels = self.all_gather(labels_local)
            all_labels = all_labels.view(-1, N_GENES)
        else:
            all_labels = None

        local_pert_ids = list(self._test_pert_ids)
        local_symbols = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        gathered_pert_ids = [local_pert_ids]
        gathered_symbols = [local_symbols]
        if world_size > 1:
            obj_pert = [None] * world_size
            obj_sym = [None] * world_size
            dist.all_gather_object(obj_pert, local_pert_ids)
            dist.all_gather_object(obj_sym, local_symbols)
            gathered_pert_ids = obj_pert
            gathered_symbols = obj_sym

        if self.trainer.is_global_zero:
            all_pert_ids = [p for rank_list in gathered_pert_ids for p in rank_list]
            all_symbols = [s for rank_list in gathered_symbols for s in rank_list]
            all_preds_np = all_preds.float().cpu().numpy()

            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            dedup_indices = []
            for i, pid in enumerate(all_pert_ids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_ids.append(pid)
                    dedup_syms.append(all_symbols[i])
                    dedup_preds.append(all_preds_np[i])
                    dedup_indices.append(i)

            dedup_preds_np = np.stack(dedup_preds, axis=0)

            if all_labels is not None:
                all_labels_np = all_labels.cpu().numpy()
                dedup_labels_np = all_labels_np[np.array(dedup_indices)]
                test_f1 = _compute_per_gene_f1(dedup_preds_np, dedup_labels_np)
                self.log("test/f1", test_f1, prog_bar=True, rank_zero_only=True)

            _save_test_predictions(
                pert_ids=dedup_ids,
                symbols=dedup_syms,
                preds=dedup_preds_np,
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

    def configure_optimizers(self):
        """Muon + AdamW dual optimizer with CosineAnnealingWarmRestarts.

        Proven recipe from node3-3-1-2-1-1-1 (F1=0.5243):
        - Muon (lr=0.01) for fc1/fc2 weight matrices in hidden residual blocks
        - AdamW (lr=3e-4) for all other parameters
        - CosineAnnealingWarmRestarts (T_0=80, T_mult=2): enables escape from local minima
          via periodic LR restarts. Proven to push F1 from 0.49 to 0.52+ via multiple cycles.

        Note: Muon is NOT applied to:
        - ESM2/STRING projection layers (first layers)
        - Output projection (last layer)
        - LayerNorm parameters
        - Biases (1D)
        - Gate network weights (keep with AdamW for stable gating)
        """
        try:
            from muon import MuonWithAuxAdam
        except ImportError:
            raise ImportError(
                "Muon optimizer not found. Install with: "
                "pip install git+https://github.com/KellerJordan/Muon"
            )

        muon_params = []
        adamw_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Apply Muon only to 2D weight matrices in hidden residual blocks
            # NOT to projection layers, NOT to output head, NOT to gate network
            is_muon_eligible = (
                param.ndim >= 2
                and "blocks." in name
                and (".fc1.weight" in name or ".fc2.weight" in name)
            )
            if is_muon_eligible:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        self.print(
            f"Muon params: {sum(p.numel() for p in muon_params):,} | "
            f"AdamW params: {sum(p.numel() for p in adamw_params):,}"
        )

        param_groups = [
            dict(
                params=muon_params,
                use_muon=True,
                lr=self.hparams.muon_lr,    # 0.01
                weight_decay=self.hparams.weight_decay,
                momentum=0.95,
            ),
            dict(
                params=adamw_params,
                use_muon=False,
                lr=self.hparams.lr,         # 3e-4
                betas=(0.9, 0.95),
                weight_decay=self.hparams.weight_decay,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # CosineAnnealingWarmRestarts: proven to help escape local minima
        # T_0=80: proven in node3-3-1-2-1-1-1 (F1=0.5243), node3-1-1-1-1-2-1-1 (F1=0.5265)
        # T_mult=2: doubles period after each restart (80→160→320)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.t0,
            T_mult=self.hparams.t_mult,
            eta_min=self.hparams.eta_min,
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
        self.print(
            f"Saving checkpoint: {trainable}/{total} params "
            f"({100 * trainable / total:.2f}%)"
        )
        return result

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-F1 averaged over all response genes.

    Matches data/calc_metric.py logic exactly:
    - argmax over class dim to get hard predictions
    - per-gene F1 averaged over present classes only
    - final F1 = mean over all genes
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)     # [N, N_GENES]
    n_genes = labels.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels[:, g]
        yh = y_hat[:, g]
        per_class_f1 = sk_f1(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        if present.any():
            f1_vals.append(float(per_class_f1[present].mean()))
        else:
            f1_vals.append(0.0)
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert len(pert_ids) == len(preds), (
        f"Length mismatch: {len(pert_ids)} ids vs {len(preds)} predictions"
    )
    rows = []
    for i in range(len(pert_ids)):
        rows.append({
            "idx": pert_ids[i],
            "input": symbols[i],
            "prediction": json.dumps(preds[i].tolist()),
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}", flush=True)


def _ensemble_checkpoints(
    checkpoint_paths: List[Path],
    model: PerturbModule,
    datamodule: PerturbDataModule,
    trainer: pl.Trainer,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Ensemble top-K checkpoints by averaging logits in prediction space.

    Proven strategy: node3-1-1-1-1-2-1-1-1 used top-3 threshold ensemble to
    achieve F1=0.5283 (+0.0018 over top-7 ensemble at F1=0.5265).
    Top-3 outperforms top-7 because fewer high-quality checkpoints beat larger
    diverse ensembles that include degraded checkpoints.

    Filtering: only include checkpoints within ensemble_threshold of the best val/f1.
    """
    if not checkpoint_paths:
        print("No checkpoints for ensemble; skipping.", flush=True)
        return

    print(f"Ensembling {len(checkpoint_paths)} checkpoints...", flush=True)
    all_preds_list = []

    for ckpt_path in checkpoint_paths:
        print(f"  Loading: {ckpt_path.name}", flush=True)
        loaded = PerturbModule.load_from_checkpoint(
            str(ckpt_path),
            strict=False,
            esm2_dim=args.esm2_dim,
            string_dim=args.string_dim,
            fusion_dim=args.fusion_dim,
            hidden_dim=args.hidden_dim,
            n_genes=N_GENES,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            head_dropout=args.head_dropout,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            mixup_prob=args.mixup_prob,
            t0=args.t0,
            t_mult=args.t_mult,
            eta_min=args.eta_min,
            max_epochs=args.max_epochs,
        )
        loaded.eval()

        # Run test on this checkpoint
        single_preds = []
        single_pert_ids = []
        single_symbols = []

        test_loader = datamodule.test_dataloader()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded = loaded.to(device)

        with torch.no_grad():
            for batch in test_loader:
                esm2_feats = batch["esm2_features"].to(device).float()
                string_feats = batch["string_features"].to(device).float()
                logits = loaded.model(esm2_feats, string_feats)
                single_preds.append(logits.cpu().float())
                single_pert_ids.extend(batch["pert_id"])
                single_symbols.extend(batch["symbol"])

        preds_tensor = torch.cat(single_preds, dim=0).numpy()  # [N, 3, N_GENES]
        all_preds_list.append((preds_tensor, single_pert_ids, single_symbols))

        del loaded
        torch.cuda.empty_cache()

    # Average logits across checkpoints (logit-space averaging = equivalent to probability averaging)
    avg_preds = np.mean([p[0] for p in all_preds_list], axis=0)
    pert_ids = all_preds_list[0][1]
    symbols = all_preds_list[0][2]

    # Deduplicate
    seen: set = set()
    dedup_ids, dedup_syms, dedup_preds_list = [], [], []
    for i, pid in enumerate(pert_ids):
        if pid not in seen:
            seen.add(pid)
            dedup_ids.append(pid)
            dedup_syms.append(symbols[i])
            dedup_preds_list.append(avg_preds[i])

    ensemble_preds = np.stack(dedup_preds_list, axis=0)

    # Save ensemble predictions
    ensemble_path = output_dir / "test_predictions.tsv"
    _save_test_predictions(dedup_ids, dedup_syms, ensemble_preds, ensemble_path)
    print(f"Ensemble predictions saved → {ensemble_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ESM2-650M + STRING_GNN Dual-Branch + Gated Fusion + Manifold Mixup"
    )
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--esm2-dim", type=int, default=ESM2_EMB_DIM)
    p.add_argument("--string-dim", type=int, default=STRING_EMB_DIM)
    p.add_argument("--fusion-dim", type=int, default=FUSION_DIM)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--head-dropout", type=float, default=0.15)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--mixup-prob", type=float, default=0.65)
    p.add_argument("--t0", type=int, default=80)
    p.add_argument("--t-mult", type=int, default=2)
    p.add_argument("--eta-min", type=float, default=1e-6)
    p.add_argument("--grad-clip-val", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=80)
    p.add_argument("--ensemble-top-k", type=int, default=3)
    p.add_argument("--ensemble-threshold", type=float, default=0.003)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--esm2-cache-path", type=str, default=None,
                   help="Path to cache precomputed ESM2 embeddings (avoids re-fetching)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default ESM2 cache to node's run directory for reuse
    if args.esm2_cache_path is None:
        args.esm2_cache_path = str(output_dir / "esm2_embeddings_cache.pt")

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        esm2_cache_path=args.esm2_cache_path,
    )

    model = PerturbModule(
        esm2_dim=args.esm2_dim,
        string_dim=args.string_dim,
        fusion_dim=args.fusion_dim,
        hidden_dim=args.hidden_dim,
        n_genes=N_GENES,
        n_blocks=args.n_blocks,
        lr=args.lr,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        t0=args.t0,
        t_mult=args.t_mult,
        eta_min=args.eta_min,
        max_epochs=args.max_epochs,
        ensemble_top_k=args.ensemble_top_k,
        ensemble_threshold=args.ensemble_threshold,
    )

    fast_dev_run = args.fast_dev_run
    debug_max_step = args.debug_max_step
    if debug_max_step is not None:
        limit_train = limit_val = limit_test = debug_max_step
        max_steps = debug_max_step
        val_check_interval = 1.0
        num_sanity_val_steps = 0
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1
        val_check_interval = args.val_check_interval if not fast_dev_run else 1.0
        num_sanity_val_steps = 2

    # Top-K checkpoint saving with threshold filter (from node3-1-1-1-1-2-1-1-1 recipe)
    # NOTE: In DDP mode, val_f1 is only available on rank 0 at checkpoint save time;
    # non-zero ranks default to 0.0 in the metrics dict. We avoid this by NOT
    # including val_f1 in the filename, and instead extract it from checkpoint
    # metadata during ensemble selection (via torch.load on rank 0 only).
    # Also: 'val/f1' as monitor key with auto_insert_metric_name=True causes
    # colons ('/') to become path separators on Linux.
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename=f"best-epoch={{epoch:03d}}",
        monitor="val/f1",
        mode="max",
        save_top_k=args.ensemble_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/f1",
        mode="max",
        patience=args.early_stop_patience,
        min_delta=1e-5,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(
            find_unused_parameters=False, timeout=timedelta(seconds=120)
        ),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=args.grad_clip_val,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, datamodule=datamodule)

    # Standard single-model test with best checkpoint
    if fast_dev_run or debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Top-K checkpoint ensemble (only on global rank 0, after training)
    # Load and ensemble the top-K checkpoints to produce the final test_predictions.tsv
    if trainer.is_global_zero and not fast_dev_run and debug_max_step is None:
        ckpt_dir = Path(output_dir / "checkpoints")
        ckpt_files = sorted(ckpt_dir.glob("best-*.ckpt"), key=lambda p: p.name)

        if len(ckpt_files) >= 2:
            # Extract val_f1 from checkpoint metadata (torch.load on rank 0 only).
            # In DDP mode, val_f1 is only logged on rank 0, so we must read it from
            # the checkpoint file's 'callbacks' section rather than the filename.
            def parse_f1_from_ckpt(path: Path) -> float:
                try:
                    ckpt = torch.load(path, map_location="cpu")
                    # Lightning stores callback metrics in the 'callbacks' key of the checkpoint
                    callbacks = ckpt.get("callbacks", {})
                    mc_state = callbacks.get("ModelCheckpoint", {})
                    best_score = mc_state.get("best_model_score", torch.tensor(0.0))
                    return float(best_score)
                except Exception:
                    return 0.0

            scored = [(parse_f1_from_ckpt(p), p) for p in ckpt_files]
            scored.sort(key=lambda x: x[0], reverse=True)
            best_f1 = scored[0][0]

            # DDP fallback: if all F1s are 0 (best_model_score not in checkpoint
            # state_dict — it's a callback attribute, not synced into checkpoint),
            # use trainer.checkpoint_callback.best_model_path (correctly DDP-broadcast)
            # to identify the best epoch, then rank by proximity to best epoch.
            if best_f1 <= 0.0 and hasattr(trainer, "checkpoint_callback") and trainer.checkpoint_callback is not None:
                best_path = Path(trainer.checkpoint_callback.best_model_path)
                try:
                    best_ckpt_epoch = int(best_path.stem.split("=")[1])
                except Exception:
                    best_ckpt_epoch = -1
                if best_ckpt_epoch >= 0:
                    scored = []
                    for p in ckpt_files:
                        try:
                            ckpt_epoch = int(p.stem.split("=")[1])
                        except Exception:
                            ckpt_epoch = -1
                        # Closer to best epoch = better (negative distance = higher score)
                        scored.append((-abs(ckpt_epoch - best_ckpt_epoch), p))
                    scored.sort()  # most negative = closest to best
                    best_f1 = 0.0  # F1 unknown but > 0

            # Filter: keep only checkpoints within ensemble_threshold of best
            selected = [
                p for f1, p in scored
                if (best_f1 - f1) <= args.ensemble_threshold
            ][:args.ensemble_top_k]

            if len(selected) >= 2:
                print(
                    f"Running top-{len(selected)} ensemble "
                    f"(threshold={args.ensemble_threshold}, best_f1={best_f1:.4f})",
                    flush=True,
                )
                # Setup datamodule for ensemble inference
                datamodule.setup(stage="test")
                _ensemble_checkpoints(
                    checkpoint_paths=selected,
                    model=model,
                    datamodule=datamodule,
                    trainer=trainer,
                    output_dir=output_dir,
                    args=args,
                )
            else:
                print(
                    f"Only 1 checkpoint within threshold; using single-model predictions.",
                    flush=True,
                )

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        result = test_results[0]
        primary_metric = result.get("test/f1", result.get("test/f1_score", float("nan")))
        score_path.write_text(str(float(primary_metric)))
        print(f"Test results → {score_path} (f1_score={primary_metric})", flush=True)


if __name__ == "__main__":
    main()
