**Improvements from Parent**
- Reduced dropout from 0.5 → 0.45
- Lowered learning rate from 1e-4 → 8e-5
- Extended early stopping patience from 15 → 20 epochs
- Added top-3 checkpoint ensemble (tested but produced zero improvement)

**Results & Metrics (vs Parent)**
- Test F1: 0.5110 (+0.0032 over parent 0.5078, -0.0019 below tree best 0.5128)
- Training: 128 epochs (vs 88), best val F1=0.5110 at epoch 107
- Train-val gap: ~0.24 (vs ~0.14 parent)
- Three checkpoints saved with val F1: 0.5100, 0.5110, 0.5105 (nearly identical)

**Key Issues**
- Checkpoint ensemble produced zero improvement due to lack of diversity within saved checkpoints
- Concatenation fusion architecture reached performance ceiling at ~0.511 F1
- 0.005 gap to tree best (node2-1-1-1-1-1, F1=0.5128) requires architectural innovation not hyperparameter tuning
