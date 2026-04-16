**Improvements from Parent**
- Reduced hidden dimensions from 512 to 384 in 3-block PreNorm Residual MLP
- Changed output head from factorized (512→256→19920) to flat projection (384→19920)
- Removed ESM2 embeddings (STRING-only frozen 256-dim)
- Replaced AdamW-only with hybrid optimizer: Muon (LR=0.01) + AdamW (LR=3e-4)
- Replaced ReduceLROnPlateau with CosineAnnealingWarmRestarts (T_0=80, T_mult=2)
- Added Manifold Mixup (prob=0.5, alpha=0.2 in hidden space)
- Added head dropout=0.15
- Removed label smoothing
- Extended training from 83 epochs (early-stopped) to 500 epochs (no early stopping)
- Set weight_decay=8e-4

**Results & Metrics (vs Parent)**
- Test F1: 0.4950 (+0.032 improvement over parent's 0.463)
- Val/F1 improvement trajectory: 0.463 at epoch 80 → 0.492 at epoch ~468
- Training completed all 500 epochs without early stopping
- Val/loss remained elevated at 1.17-1.22 after each warm restart
- Model ranked second-best in STRING-only lineage

**Key Issues**
- Insufficient training duration: val/F1 still climbing at epoch 500, model had not converged
- Elevated validation loss (1.17-1.22) persisted throughout warm restart cycles
- Destructive warm restart frequency (T_0=80) interrupted optimization momentum
- Single checkpoint evaluation (no ensemble) limited final test performance
