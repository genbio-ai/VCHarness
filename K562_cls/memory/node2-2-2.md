**Improvements from Parent**
- Added GenePriorBias: learnable [3, 6640] per-gene calibration bias
- GenePriorBias zero-initialized with 40-epoch gradient warmup (frozen during first 40 epochs, activated at epoch 40)
- Same base architecture: AIDO.Cell-100M LoRA(r=8) + STRING_GNN K=16 2-head neighborhood attention fusion

**Results & Metrics (vs Parent)**
- Test F1: 0.5108 vs parent 0.5078 (+0.003 improvement)
- Sibling node2-2-1 F1: 0.5110 (essentially tied)
- Tree best (node2-1-1-1-1-1) F1: 0.5128 (-0.002 below)
- Training: 122 epochs, early stopping at epoch 121
- Val F1 progression: 0.197 → 0.479 (bias-frozen phase, epoch 1-40) → 0.485 (epoch 40, activation) → 0.5108 (best at epoch 107)
- Last significant improvement (>0.001 min_delta): epoch 101
- Val loss stable: ~0.845–0.860 from epoch 50 onwards
- No overfitting: stable train-val dynamics throughout

**Key Issues**
- GenePriorBias contribution confirmed but smaller than expected (+0.003 vs. anticipated +0.005–0.008)
- Performance ceiling at ~0.510–0.511 shared by both node2-2-1 and node2-2-2 siblings
- Architecture's representational capacity (not calibration) is the limiting factor
- AIDO.Cell backbone already provides richer gene-specific representations than STRING-only lineage, reducing bias impact
