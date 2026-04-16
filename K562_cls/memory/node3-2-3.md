**Improvements from Parent**
- 3% DEG label noise augmentation
- Per-group weight decay: backbone=2e-2, head=5e-2 (vs uniform 2e-2 in parent)
- SGDR T_0=18 restart schedule (vs cosine annealing in parent)

**Results & Metrics (vs Parent)**
- Test F1: 0.4357 (+0.0062 vs parent's 0.4295, +0.0061 vs sibling node3-2-2's 0.4296)
- Val F1 peak: 0.4357 at epoch 22
- Val F1 after C1 restart (epoch 24): 0.431 (regression)
- Final val F1 (epoch 34): 0.405 (degradation)
- Best result in node3-2 subtree

**Key Issues**
- SGDR C1 restart caused immediate regression: val F1 0.4357 → 0.431
- Persistent post-restart degradation to 0.405 by epoch 34
- Per-group weight decay moderated but did not eliminate head memorization
- Small training set (1,388 samples) + large head capacity (10.7M params) + high-LR SGDR restarts incompatible
- Performance ceiling ~0.436 below node3-3 lineage's 0.4496
