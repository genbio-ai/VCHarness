**Improvements from Parent**
- Removed focal loss (γ=2.0) → standard weighted cross-entropy
- Dropped ESM2 embeddings → reverted to 256-dim STRING_GNN-only (from 736-dim)
- Reduced MLP depth: 4 blocks (parent had 3)
- Added 5-epoch linear LR warmup
- Lowered LR: 2e-4 (parent: 5e-4)
- Changed T_max: 80 epochs (parent: 100)
- Increased dropout: 0.40 (parent: implied lower)
- Adjusted weight decay: 0.015 (parent: 0.01)
- Added gradient clipping + accumulation (effective batch=512)

**Results & Metrics (vs Parent)**
- Test F1: 0.336 vs parent 0.157 (+114% relative improvement)
- Val F1 epoch 0: 0.270 vs parent 0.1756
- Val F1 collapse epochs 0-5: 0.270→0.195 (-28%)
- Peak val F1 at epoch 62 (vs node1-1 at epoch 28 = 34 epochs delayed)
- Final val/loss plateau: 1.177 with training loss still declining
- Performance gap vs best node1-1: -29% (0.336 vs 0.472)

**Key Issues**
- LR warmup caused severe early F1 collapse (28% drop epochs 0-5)
- Convergence delayed by 34 epochs compared to node1-1
- Val/loss plateau at 1.177 while training loss declines → overfitting
- Excessive regularization combo: dropout 0.40 + wd 0.015 + clipping + low LR 2e-4
- Reduced model capacity (4 blocks) below proven node1-1 architecture (5 blocks)
