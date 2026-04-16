**Improvements from Parent**
- Dual fusion concatenating frozen STRING PPI embeddings (256-dim) + ESM2 protein embeddings (480-dim) → 736-dim input
- 3 residual MLP blocks with per-gene bias, projecting to 512-dim
- Weighted cross-entropy + label_smoothing=0.05
- AdamW optimizer (lr=3e-4, wd=5e-4)
- ReduceLROnPlateau scheduler (patience=5)

**Results & Metrics (vs Parent)**
- Test F1=0.444 (regression -0.030 vs parent node1-1-1's 0.474)
- Test F1=0.444 (regression -0.011 vs focal-loss predecessor's 0.455)
- Train loss plateaued at 0.240 (20× higher than parent's 0.012)
- Val/F1 peaked at epoch 20 (0.444), then oscillated ±0.02 through epoch 45
- ReduceLROnPlateau fired 4 LR halvings starting epoch 26 with zero recovery

**Key Issues**
- Severe underfitting: train loss stuck at 0.240 across all 45 epochs
- ESM2 fusion degrades input representation; 736-dim concatenated input creates harder optimization landscape than 256-dim STRING-only
- 3-block MLP insufficient to navigate the concatenated embedding space
- LR reductions applied too late (epoch 26+) after model already entered suboptimal basin at epoch 20
