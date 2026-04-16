**Improvements from Parent**
- Replaced AdamW-only optimizer with MuonWithAuxAdam dual optimizer (Muon lr=0.01 for 2D MLP trunk matrices, AdamW lr=3e-4/wd=8e-4 for other params)
- Changed from CosineAnnealingLR (T_max=200) to CosineAnnealingWarmRestarts (T_0=80, T_mult=2)
- Added Manifold Mixup (alpha=0.2, prob=0.5 at embedding level)
- Reduced hidden dimensions from 512→1024→512 to 384→768 with ~9.6M trainable params (vs parent's ~9.6M)
- Increased max_epochs from 200 to 500, patience from 25 to 160

**Results & Metrics (vs Parent)**
- Test F1: **0.4773** vs parent 0.3773 (+0.100)
- Best val F1: 0.4773 at epoch 339
- Early stopping triggered at epoch 499 (patience=160 from epoch 339)
- CosineWarmRestarts measurable gains: +0.008 F1 at epoch 80, +0.011 F1 at epoch 240
- Loss-space overfitting: val/train loss ratio = 6.55× at final epoch; val loss rose 84.7% from epoch-22 minimum

**Key Issues**
- Third CosineWarmRestarts cycle truncated by early stopping at epoch 499 (cycle 3 started at epoch 240, next restart would be at epoch 560)
- Val F1 remained robust despite significant loss-space overfitting (val/train loss ratio 6.55×)
- ~0.020 F1 gap to tree-best STRING-only node (F1=0.4968) likely due to incomplete cycle 3 rather than architectural insufficiency
