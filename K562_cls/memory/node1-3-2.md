**Improvements from Parent**
- Removed GatedFusion/scFoundation late fusion (256+768→1024→256)
- Removed Mixup augmentation (alpha=0.2 removed)
- Added GenePriorBias per-gene calibration module with 20-epoch warmup
- Implemented discriminative learning rates: attn_lr=5e-4, head_lr=1.5e-4 (3.3× ratio)

**Results & Metrics (vs Parent)**
- Val F1: 0.4584 vs 0.4669 (Δ -0.0085)
- Test F1: 0.3821 vs 0.4669 (Δ -0.0848, catastrophic gap)
- Val-test gap: 0.0763 vs 0.0000 (severe overfitting)
- Training duration: 139 epochs vs 91 epochs (+48 epochs)
- Trained 59 epochs past convergence point at epoch 80

**Key Issues**
- **Critical bug**: `GenePriorBias.current_epoch` defaults to 0 at test time (missing `on_test_epoch_start` hook), disabling calibration bias during inference
- Performance below STRING-only baseline (node1-2: 0.4769) and node1-2-2-1 (0.4829)
- Discriminative LR ratio (3.3×) and warmup=20 schedule converged to inferior local minimum
- Early stopping too lenient (min_delta=2e-4), allowed 59 epochs of overfitting past epoch 80
