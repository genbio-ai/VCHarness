**Improvements from Parent**
- WarmupCosine LR schedule (10-epoch warmup, 150-epoch cosine decay to 5%) replacing ReduceLROnPlateau
- Increased regularization: weight_decay 1e-2→2e-2, head_dropout 0.3→0.5, added fusion_dropout=0.2
- Added label smoothing to focal loss
- Discriminative learning rates for GNN branch

**Results & Metrics (vs Parent)**
- Test F1: 0.4629 vs 0.4585 (+0.96% absolute gain)
- Train/val loss gap: 12x vs 18x (parent), reduced by 33%
- Generalization gap: near-zero test/val gap
- Training epochs: 101, best checkpoint at epoch 80
- Val F1 oscillation: 0.44–0.46 band from epoch 25 onward
- Val loss increase: 0.054→0.089 over training

**Key Issues**
- Cosine annealing floor (min_lr_ratio=0.05) leaves LR=1e-5 in later epochs with insufficient exploration
- 10.2M-parameter linear classification head exhibits memorization (12x train/val loss gap)
- Compound focal+label-smoothing loss poorly correlates with F1 metric
- Val F1 plateau with persistent oscillation, val loss steadily increases
