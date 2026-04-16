**Improvements from Parent**
- Switched from AIDO.Cell-10M direct QKV fine-tuning (22.26M params, 70.23% trainable) to AIDO.Cell-100M with LoRA (rank=16, alpha=32) for parameter-efficient adaptation
- Added triple-stream architecture: (A) perturbed gene position-specific AIDO.Cell embedding, (B) mean-pooled global transcriptome state, (C) frozen STRING_GNN PPI embeddings (256-dim)
- Enhanced input encoding with full multi-gene baseline context (all 19,264 genes at 1.0, perturbed gene at 0.0)
- Changed from weighted cross-entropy to focal loss (gamma=2.0) with same class weights [10.91, 1.0, 29.62]
- Replaced Muon optimizer with AdamW (backbone lr=1e-4, head lr=3e-4)
- Used LayerNorm→Linear(1024)→GELU→Dropout(0.2)→Linear(6640×3) fusion head

**Results & Metrics (vs Parent)**
- Test F1: **0.4579** (+0.0726 vs node3-1's 0.3853; +0.0431 vs node3-1-2's 0.4148)
- Best val F1: 0.4578 at epoch 50 (out of 91 total epochs, early stopping patience=40)
- Best node3-1-x lineage result; converged after epoch 50 with val F1 plateauing at 0.44–0.46 (mean 0.4487)
- Training loss: smooth decrease 0.0313→0.0084 across 90 epochs
- Validation loss: decreased to ~0.026 at epoch 10, rose to ~0.031 by epoch 50 (~3× gap vs training)

**Key Issues**
- Frozen STRING_GNN embeddings cannot adapt to hTERT-RPE1 cell-line context (no learnable adapter)
- MLP fusion head lacks gene-specific output projections (node1-1-3-1-1 achieved 0.4858 with bilinear head)
- Model reached capacity ceiling under current architecture after epoch 50
- Validation overfitting: val loss diverged from training loss by ~3× at epoch 50
