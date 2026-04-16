**Improvements from Parent**
- Replaced AIDO.Cell-100M synthetic expression encoding with dual-source frozen biological embeddings: STRING_GNN PPI graph (256-dim) + ESM2-35M protein sequence (480-dim) concatenated to 736-dim per perturbed gene
- Changed model architecture from mean-pooled AIDO.Cell encoding to 3-block residual MLP head (512-dim hidden, ~13.7M params)
- Replaced weighted cross-entropy with focal loss (γ=2.0) + weighted CE + label smoothing (0.05)
- Changed optimizer from AdamW with default settings to AdamW (lr=5e-4, wd=0.01) with cosine annealing (T_max=100)

**Results & Metrics (vs Parent)**
- Test F1: 0.157 (identical to parent node3's 0.157)
- Far below best sibling node1-1 (STRING_GNN-only, weighted CE, F1=0.472)
- Training loss: 0.065→0.009 (continuous decrease)
- Val F1 collapsed from 0.1756 (epoch 0) to 0.0592 (epoch 1) — 66% drop, then partial recovery to ~0.096 by epoch 26
- Model predicts all-neutral (matches naive baseline for 92.8% class-0 distribution)

**Key Issues**
- Focal loss with γ=2.0 down-weights majority neutral class (92.8%) too aggressively, destabilizing optimizer and driving degenerate all-neutral predictions
- ESM2 embeddings (480-dim) dilute STRING_GNN signal (256-dim) — concatenation produces 736-dim vector with noise rather than signal
- Frozen embeddings cannot adapt to the specific perturbation response prediction task
- Early training instability: val/f1 collapses immediately after epoch 0 despite continuous training loss decrease
