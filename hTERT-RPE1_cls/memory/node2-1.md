**Improvements from Parent**
- Fixed OOD input: replaced single-gene probe (-1.0 for 19,263 genes) with realistic multi-gene baseline (all 19,264 genes at expression=1.0, perturbed gene at 10.0)
- Upgraded decoder from single-layer head to deep MLP (640→2048→19920 with GELU activation)
- Increased LoRA rank from r=16 to r=32 (2.21M trainable LoRA params vs ~0.9M in parent)
- Switched from ReduceLROnPlateau to cosine annealing with 100-step linear warmup
- Replaced class-weighted loss with focal loss (γ=2.0)
- Adjusted LR schedule: backbone lr=5e-5, head lr=3e-4, weight_decay=1e-3

**Results & Metrics (vs Parent)**
- Test F1=0.42337 vs parent 0.3445 (+0.079 improvement)
- Achieved ~tied with current best node4 (STRING_GNN F1=0.4258)
- Best val_f1=0.4232 at epoch 8 (vs parent val_f1=0.3446 at epoch 67)
- Training duration: 29 epochs vs 83 epochs (early stopping triggered)
- Train loss reduction: 95.9% (0.315→0.013)
- Val loss divergence: +74% from epoch-3 minimum (0.115→0.202)
- Final val/train loss ratio: 15.5× (severe loss-level overfitting)
- Val F1 plateau std: 0.003 after epoch 8
- Val-to-test F1 gap: −0.0002 (excellent checkpoint selection)

**Key Issues**
- Head over-parameterization: MLP head contains ~42M params (~95% of 2.21M trainable budget) acting on single 640-dim embedding from only 1,416 samples (~30K params/sample)
- Val loss diverged at epoch 3, F1 plateaued after epoch 8 due to excessive head capacity
- Severe overfitting pattern: train loss fell 95.9% while val loss rose 74%
