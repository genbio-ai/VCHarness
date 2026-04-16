**Improvements from Parent**
- Replaced 5-block residual MLP with 3-block residual MLP (hidden=512)
- Added multi-scale STRING_GNN (768-dim: concat layers 4+6+8 projected to 512)
- Introduced factorized output head (bottleneck=256)
- Switched from focal loss to hybrid focal + WCE loss (50/50)
- Changed from cosine annealing to ReduceLROnPlateau scheduler

**Results & Metrics (vs Parent)**
- Test F1: 0.4357 vs parent 0.4555 (−0.020 regression)
- Test F1 vs tree-best: 0.4357 vs node1-1-1's 0.474 (−0.038)
- Training epochs: 78, best checkpoint at epoch 47 (val/F1=0.436)
- Train loss: 0.514 (stuck, vs node1-1-1's 0.012)
- Val loss: 0.636 (vs node1-1-1's 0.040)

**Key Issues**
- Severe underfitting: model cannot fit training data (train/loss stuck at 0.514)
- Double information bottleneck: 768→512 compression before MLP + 512→256 bottleneck in output head
- Multi-scale concatenation diluted proven final-layer signal
- Hybrid loss prevented confident predictions
