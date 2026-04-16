**Improvements from Parent**
- Replaced learned gene embedding table with frozen STRING_GNN pretrained PPI-graph embedding (256-dim)
- Changed loss from weighted cross-entropy + label smoothing to focal loss
- Added cosine annealing learning rate schedule
- Reduced MLP head from 8 to 5 residual blocks
- Reduced trainable parameters from 19.4M to 15.6M

**Results & Metrics (vs Parent)**
- Test F1: 0.472 vs 0.405 (+0.067, +16.5% relative improvement)
- Best val F1: 0.472 at epoch 28
- Final val F1: 0.467 at epoch 53
- Training loss: 0.078 → 0.015 (5.2× reduction)
- Val loss: 0.040 → 0.057 (1.4× increase)
- LR decay: 3e-4 → 2.7e-4 over 53 epochs

**Key Issues**
- Fundamental information bottleneck: single perturbed gene ID → 6,640 gene responses
- Excessive model capacity: 15.6M trainable params for 1,273 training samples
- Clear overfitting: training loss decreased while val loss increased
- Cosine annealing schedule too slow: minimal LR decay over 53 epochs
- Insufficient regularization to prevent overfitting
