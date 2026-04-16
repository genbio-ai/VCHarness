**Improvements from Parent**
- Increased MLP hidden_dim from 512 → 640
- Increased residual blocks from 3 → 4
- Parameters increased from ~13M → 19.5M
- Loss changed from focal loss → weighted cross-entropy + label smoothing
- LR scheduler changed from cosine annealing → ReduceLROnPlateau (lr=2e-4)
- Added per-gene bias initialization
- Dropout increased to 0.40

**Results & Metrics (vs Parent)**
- Test F1: 0.4297 (parent: 0.4717, -0.0420)
- Best val F1: 0.4803 at epoch 34
- Val-test gap: 0.0506 (ranked last among all relatives)
- Training dynamics: val loss increased from epoch 3 onward (1.235→1.217), train loss declined (1.540→0.982)
- Aggressive 4× LR reduction triggered during training

**Key Issues**
- Catastrophic overfitting: 0.0506 val-test gap, worst among all sibling nodes
- Increasing hidden_dim to 640 with 19.5M params on 1,273 samples was counterproductive
- Switching from focal loss to weighted CE + label smoothing degraded performance
- Validation loss plateaued/increased while training loss steadily declined
- Wider representation increased fitting capacity without improving generalization
