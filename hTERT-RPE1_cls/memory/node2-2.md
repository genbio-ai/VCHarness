**Improvements from Parent**
- Increased LoRA rank from r=16 to r=64 (4.42M backbone params vs ~0.9% trainable)
- Replaced single-layer linear head (640→19920) with bilinear interaction head (2.4M params): 640→512→768 MLP projector + 256-dim × [6640] output gene embeddings
- Changed input from synthetic probe (perturbed=1.0, others=-1.0) to multi-gene expression (all-1.0 baseline, perturbed=10.0)
- Switched from class-weighted loss to focal loss (γ=2.0) with label smoothing (ε=0.05)
- Replaced ReduceLROnPlateau LR schedule with cosine schedule (correctly calibrated, reached zero at epoch 61)

**Results & Metrics (vs Parent)**
- Test F1: 0.4102 vs 0.3445 (+19.1% improvement)
- Best val F1: 0.4100 at epoch 37 (vs parent 0.3446 at epoch 67)
- Convergence: faster early improvement (val F1 0.365→0.403 through epoch 10), plateau 0.40–0.41 from epoch 22 onward
- Overfitting: val/train loss ratio 2.0× at epoch 62 (vs parent divergence pattern)
- Training duration: 62 epochs, early stopped (vs 83 epochs)

**Key Issues**
- Bilinear interaction dimension (256) too small for distinguishing perturbation effects across 6,640 genes from 640-dim AIDO.Cell embeddings
- Performance gap: 3.1% below sibling node2-1 (F1=0.4234), 16.4% below tree-best node1-2 (F1=0.4912)
- Val loss increased from minimum 0.121 (epoch 3) to 0.138 (epoch 62), indicating mild overfitting
- Focal loss may benefit from explicit minority-class weighting for non-neutral gene positions
