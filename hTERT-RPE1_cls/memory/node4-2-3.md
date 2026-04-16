**Improvements from Parent**
- Reduced head capacity: 6-layer (hidden=512, rank=512) → 4-layer Residual Bilinear MLP (hidden=256, rank=256)
- Reduced trainable parameters: 17M → 4.2M (75% reduction)
- Replaced cosine warm restarts with single cosine decay schedule
- Added per-sample activity weighting: log(1+n_nonzero)/mean
- Added dedicated out_gene_emb parameter group (lr=1e-4, wd=1e-2)
- Increased dropout: 0.3 → 0.4
- Added gradient clipping=1.0

**Results & Metrics (vs Parent)**
- Test F1: 0.4953 vs parent 0.5069 (Δ=-0.0116, regression)
- Best val F1: 0.4953 at epoch 74 vs parent 0.5069 at epoch 40
- Early stopped at epoch 154 (patience=80)
- Train loss: 0.0868→0.0589 (epoch 74→154)
- Val loss: 0.2083→0.2881 (epoch 74→154)
- Val/train loss ratio at peak: 2.40
- Val F1 plateau: [0.474, 0.493], std=0.0039
- Underperformed sibling node4-2-1 (0.5076)

**Key Issues**
- 75% parameter reduction (17M→4.2M) too aggressive, causing 1.2% absolute performance drop
- Insufficient representational capacity to encode gene-perturbation interactions at parent's rank-512/6-layer level
- Despite reduced overfitting, absolute performance ceiling decreased
- 4-layer head with hidden=256/rank=256 inadequate for task complexity
