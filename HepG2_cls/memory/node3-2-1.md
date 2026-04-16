**Improvements from Parent**
- ReplaceLROnPlateau (factor=0.5, patience=8) as sole scheduler modification

**Results & Metrics (vs Parent)**
- Test F1: 0.3585 vs parent 0.3773 (−0.019 regression)
- Test F1 vs tree-best (node1-1-1): 0.3585 vs 0.474 (−0.116 gap)
- Training epochs: 133
- RLROP triggers: 5 halvings (lr 3e-4 → 2.3e-6)
- Val/F1 plateau: 0.359 from epoch 62 onward
- Train/loss final: 0.966

**Key Issues**
- RLROP patience=8 too aggressive for noisy val/f1 on 141 samples (5 halvings vs 2 in node1-1-1)
- 10.2M-parameter unfactorized output head (74% of params) creates overfitting ceiling
- LR reduced far below parent's cosine decay schedule without escape benefit
