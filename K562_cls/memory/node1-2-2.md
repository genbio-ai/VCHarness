**Improvements from Parent**
- Fused frozen pre-computed scFoundation single-gene perturbation embeddings (768-dim) with frozen STRING_GNN via 2-layer MLP fusion head (1024→512→256)
- Same bilinear gene-class head architecture as parent
- AdamW optimizer: lr=3e-4, weight_decay=3e-2, dropout=0.35
- Cosine annealing LR schedule: T_max=150

**Results & Metrics (vs Parent)**
- Test F1: **0.4558** vs parent 0.4769 (Δ=−0.021), vs best-in-tree 0.4846 (Δ=−0.029)
- Best checkpoint at epoch 71, early stopping at epoch 79
- Val F1 progression: 0.186→0.456 (slow monotonic rise)
- Train-val loss gap: 0.206 (indicates underfitting, not overfitting)
- Numerically stable training with smooth convergence

**Key Issues**
- Frozen scFoundation embeddings (pretrained on steady-state gene expression, encoding genes as 1-hot expression=1.0 vectors) provide zero perturbation-specific transcriptional signal
- scFoundation introduces orthogonal noise that degrades STRING_GNN's PPI topology information
- Fusion head underfits frozen dual-source embeddings (0.206 train-val loss gap)
- Frozen scFoundation fusion hypothesis falsified across 4 independent experiments (node1-3, node1-3-1, node1-1-2-1-1, node1-2-2), all performing below their STRING-only parents
