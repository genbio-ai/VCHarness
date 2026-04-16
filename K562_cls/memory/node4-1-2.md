**Improvements from Parent**
- Added 2-layer classification head (512→256→19920) with LayerNorm+GELU activation (5.3M params)
- Introduced Gaussian embedding noise augmentation (std=0.1) on both scFoundation and GNN embeddings
- Reduced weight_decay from 2e-2 to 1.5e-2

**Results & Metrics (vs Parent)**
- Test F1: 0.4290 vs parent 0.4629 (severe regression of -0.034)
- Training loss at best epoch: 0.074 vs parent ~0.016 (4.6x higher)
- Train/val loss gap: ~1.0x vs parent ~5x (essentially no gap, indicating underfitting)
- Best val F1: 0.033 points lower than parent
- Training ran 178 epochs, best checkpoint at epoch 152, no early stopping triggered

**Key Issues**
- Severe underfitting: minimal train/val loss gap and high training loss indicate model cannot fit training data
- Embedding noise augmentation (std=0.1) creates train/test distribution mismatch—model trains on corrupted representations but evaluated on clean ones
- 2-layer head bottleneck (512→256→19920) insufficient capacity for 1,388 training samples across 19,920 classes
- Weight reduction (1.5e-2 vs 2e-2) contributed to underfitting rather than helping regularization
