**Improvements from Parent**
- Removed partial backbone fine-tuning that destabilized PPI representations
- Returned to frozen STRING_GNN backbone with precomputed embeddings stored as buffer
- Kept low-rank additive post-embedding conditioning mechanism (pert_U [18871×16] + pert_V [16×256], ~306K params)
- Maintained 6-layer residual MLP head architecture (hidden_dim=512, bilinear rank-256 interaction, focal loss γ=2.0)

**Results & Metrics (vs Parent)**
- Test F1: **0.4714** (parent: 0.4120) → **+0.0594 improvement**
- Trainable parameters: 5.3M
- Best epoch: 89 of 140
- Initial val F1: 0.3521 at epoch 0 (matching frozen backbone baseline node1-2's 0.3510)
- Val/train loss ratio at best epoch: 2.58× (mild calibration overfitting)
- Tree best node1-2 F1: 0.4912 → this node is **−0.0198 below best**

**Key Issues**
- Transductive pert_U lookup table: per-gene offsets stored as lookup matrix
- Val/test gene rows never receive gradient updates during training (different split)
- Conditioning offsets for unseen genes remain at near-zero initialization
- Results in small noise rather than meaningful perturbation-specific signal
- Architectural mismatch between transductive design and strictly disjoint train/val/test splits
