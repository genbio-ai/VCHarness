**Improvements from Parent**
- Focal loss gamma reduced from 2.5 to 2.0
- Weight decay increased from 0.02 to 0.06
- Head dropout increased from 0.08 to 0.15
- Early stop patience reduced from 50 to 20
- CosineAnnealingWarmRestarts T_0 reduced from 80 to 40
- Manifold Mixup added (alpha=0.2, prob=0.5)
- Per-gene bias removed (19,920 params), keeping only class-level 3-param bias

**Results & Metrics (vs Parent)**
- Test F1: 0.4433 vs 0.4884 (−0.045 regression)
- Val F1: 0.4439 vs 0.5431 (−0.099 regression)
- Val-test gap: 0.0006 vs 0.0547 (improved alignment)
- Train/val loss ratio: 1.92x vs 13.04x (overfitting suppressed)
- Best checkpoint: epoch 78 vs 381
- Total epochs: 99 vs 431
- Below target range: 0.4433 vs 0.490–0.500 expected (−0.046)

**Key Issues**
- Over-regularization suppressed discriminative capacity — val_f1 plateaued at 0.44 from epoch 62 onward
- Simultaneous removal of per-gene bias (biological calibration) with aggressive Mixup (prob=0.5) and high weight decay (0.06) prevented recovery of per-gene signal
- Both val_f1 and test_f1 declined together, indicating genuine capacity loss rather than memorization reduction
