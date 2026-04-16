**Improvements from Parent**
- Changed from full STRING_GNN fine-tuning to fully frozen STRING_GNN with cached embeddings
- Increased scFoundation fine-tuned layers from 4 to 6
- Switched from focal loss to weighted CE with label_smoothing=0.1
- Added Mixup regularization (alpha=0.2)
- Replaced ReduceLROnPlateau with WarmupCosine LR schedule (min_lr_ratio=15%)

**Results & Metrics (vs Parent)**
- Test F1: 0.4801 vs 0.4585 (+0.0216)
- Train/val loss gap: ~1.25× vs 18× in parent
- Validation-test gap: 3.5e-9 (essentially zero)
- Best checkpoint at epoch 139 out of 165 total epochs
- Ranked 2nd among all MCTS nodes

**Key Issues**
- Insufficient training budget - peaked at epoch 139 but 200-epoch limit with patience=25 caused premature termination
- High LR floor (15%) potentially too aggressive for low-LR fine-tuning regime
