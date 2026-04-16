**Improvements from Parent**
- Added 2-layer classification head (512→256→19,920) instead of single linear layer
- Switched from WarmupCosine to Cosine Annealing with Warm Restarts (CAWR, T_0=40, T_mult=2)

**Results & Metrics (vs Parent)**
- Test F1: 0.3895 vs parent 0.4629 (regression of -0.073)
- Best val F1: 0.4118 at epoch 51 (vs parent 0.46+ plateau)
- Early stopping at epoch 76 (vs parent 101 epochs)
- Severe underfitting: peaked early then oscillated 0.28-0.41 after warm restart

**Key Issues**
- Insufficient capacity: frozen STRING_GNN + 4 scF layers (vs 6 in parent) unable to express task complexity
- CAWR schedule mismatched with patience=25 early stopping: model terminated before recovering from warm restarts
- Excessive regularization (weight_decay=3e-2, head_dropout=0.5, focal_gamma=2.0, label_smoothing=0.1) biased toward majority neutral class
- Warm restart at epoch 52 caused destructive LR jump to 2e-4, collapsing val F1 from 0.41 to 0.30
