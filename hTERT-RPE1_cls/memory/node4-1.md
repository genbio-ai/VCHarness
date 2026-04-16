**Improvements from Parent**
- Per-sample cond_emb injection: 256-dim all-ones vector at perturbed gene propagated through 8 GCN layers
- Multi-layer attention-weighted fusion of all 9 hidden states
- 2-layer MLP interaction head (256→512→3×256) vs parent's bilinear head
- 10× stronger weight_decay: 1e-3 vs parent's 1e-4
- CosineAnnealingLR scheduler vs parent's ReduceLROnPlateau

**Results & Metrics (vs Parent)**
- Test F1: 0.4024 vs parent 0.4258 (Δ -0.0234, regression)
- Val F1: 0.4027 at epoch 166 vs parent 0.4260 at epoch 32
- Training epochs: 191 vs parent 53
- Parameters: 7.65M vs parent 5.43M
- Train_loss: plateaued at ~0.89 vs parent 0.402
- Val_loss: increased from 1.137→1.190 vs parent 0.706→0.888
- Worst test outcome among all nodes so far

**Key Issues**
- Over-regularization from weight_decay=1e-3 prevents effective training data fitting (train_loss stalled at 0.89 across all 191 epochs)
- Cond_emb signal dilution: 256-dim all-ones vector at single node across 18,870-node graphs with 8 propagation layers produces severely diluted perturbation signal
- Simultaneous calibration overfitting (val_loss increasing) and underfitting (train_loss plateauing)
