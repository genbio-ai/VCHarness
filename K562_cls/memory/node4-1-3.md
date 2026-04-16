**Improvements from Parent**
- scFoundation: top-6 layers fine-tuned
- Classification head: expanded to 2-layer 512→1024→19920 with LayerNorm+GELU+Dropout
- Cosine annealing floor: raised min_lr_ratio from 0.05 to 0.12
- Focal loss gamma: reduced from default to 1.5
- Training epochs: extended from 150 to 200

**Results & Metrics (vs Parent)**
- Test F1: 0.4536 (vs 0.4629, -0.93%)
- Val/test alignment: exact match, no overfitting
- Epoch-75 plateau persisted despite extended training

**Key Issues**
- 2-layer head over-constrained embedding-to-logits mapping vs single linear layer
- Structural regularization reduced capacity to reach parent's peak F1
- Raised cosine floor (12%) + 200 epochs did not escape plateau
- Wider head + additional layer failed to improve representation learning
