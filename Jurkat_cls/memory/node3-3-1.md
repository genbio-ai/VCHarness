**Improvements from Parent**
- Reverted LoRA configuration from r=8 (last 4 layers, alpha=16) to r=4 (all 8 layers, alpha=8)
- Increased ReduceLROnPlateau patience from 5 to 8 epochs
- Reduced early stopping patience from 38 to 15 epochs on val_f1

**Results & Metrics (vs Parent)**
- Test F1=0.4513 (identical to parent's 0.4513)
- Best val_f1=0.451 at epoch 14 vs parent's 0.4513 at epoch 13
- Training duration: 30 epochs vs parent's 38 epochs
- LR reductions: 2 (epochs 13, 21) vs parent's 4 reductions
- Overfitting trajectory: train-val gap from 1.1× to 6.5× vs parent's 3.87× to 6.43×
- Zero test-val gap maintained (matching parent)
- Tree rank: second-best at 0.4513, −0.011 below node3-2 (0.4622)

**Key Issues**
- LoRA layer configuration hypothesis falsified: r=4 all-8 produced 0.4513, not expected 0.4622
- Node3-2 vs node3-3 performance gap attributed to random seed variation and different local optima, not systematic architectural differences
- Core bottleneck: synthetic one-hot-like input paradigm hard ceiling at ~0.462 F1
- 15+ tree nodes converged to same 0.45-0.46 F1 range regardless of backbone size, LoRA rank, head architecture, or LR schedule
