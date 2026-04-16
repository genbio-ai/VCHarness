**Improvements from Parent**
- Simplified architecture: dropped ESM2 embeddings entirely (736-dim → 256-dim STRING_GNN-only input)
- Changed network head from 3-block residual MLP (512-dim) to 3-block PreNormResNet (384-dim, dropout=0.05)
- Replaced focal loss (γ=2.0) + weighted CE + label smoothing with Manifold Mixup (alpha=0.2, prob=0.5) + standard CE
- Switched optimizer from AdamW (lr=5e-4, wd=0.01) to Muon+AdamW
- Changed scheduler from cosine annealing (T_max=100) to ReduceLROnPlateau

**Results & Metrics (vs Parent)**
- Test F1: **0.4566** vs parent 0.157 (+0.300 absolute improvement)
- Best val F1: 0.5117 at epoch 55
- Training stability: train loss smoothly decreased 1.18→0.29 over 95 epochs (no catastrophic collapse vs parent's 66% val/f1 drop epoch 0→1)
- Best result in node3-1 branch: +0.121 over best sibling

**Key Issues**
- Excessive RLROP triggering: 6 learning rate halvings vs reference node's 1-2 (patience=8 too short for natural val/f1 variance of ±0.003)
- Wasted computation: 40 epochs past peak due to early_stop_patience=40
- Val-test gap: 0.055 (0.5117→0.4566), larger than expected due to optimization instability from frequent LR halvings
- Performance gap of 0.04 vs reference node3-3-1-2 (F1=0.4966) using identical architecture but better RLROP configuration
