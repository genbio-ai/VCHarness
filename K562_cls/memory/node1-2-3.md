**Improvements from Parent**
- Added GenePriorBias output calibration with zero-initialization and gradient warmup=40 epochs
- Switched from cosine annealing LR to AdamW optimizer with lr=3e-4
- Increased dropout from parent's unspecified level to 0.40
- Added weight_decay=4e-2 and patience=25 for early stopping
- Extended training from 79 to 177 epochs (early-stopped)

**Results & Metrics (vs Parent)**
- Test F1: 0.4808 vs parent 0.4769 (+0.0039 improvement)
- Best val F1: 0.4808 at epoch 157
- Early stopping at epoch 177 (patience=25)
- Train/val loss gap: ~0.20
- Performance ceiling reached: ~0.0105 below best STRING-only node (node1-1-1-1-2-1, F1=0.4913)

**Key Issues**
- Converged to plateau at ~0.481 F1 with no further improvement
- Bilinear head's ~5.1M parameters cannot improve perturbation-to-DEG mapping given fixed STRING embeddings
- Frozen STRING_GNN backbone limits performance ceiling with current architecture
- Same architecture and hyperparameters as node1-1-1-1-2-1 but achieves 0.0105 lower F1, suggesting initialization or data ordering differences
