**Improvements from Parent**
- Switched from LoRA fine-tuning (rank=8, 6 layers) to frozen AIDO.Cell-10M backbone
- Replaced dual pooling (gene positional + global mean-pool, 512-dim) with pre-computed STRING PPI embeddings
- Expanded MLP head from 2-stage (640→256→19920) to 3-stage (896→384→256→19920, 5.5M params)
- Added differential learning rates across network components
- Changed scheduler from ReduceLROnPlateau to CosineAnnealingLR

**Results & Metrics (vs Parent)**
- Test F1: 0.4420 (+0.0076 vs parent's 0.4344)
- Best val_f1: 0.5203 at epoch 93 (vs parent's 0.5142 at epoch 43)
- Val-test gap: 0.078 (vs parent's 0.080, nearly identical)
- Train loss: 0.540→0.099 (82% reduction, more severe than parent's 62%)
- Val loss: 0.278→0.450 (62% increase, similar to parent's 67%)
- MCTS tree ranking: 4th overall (-0.020 below node3-2's 0.4622)

**Key Issues**
- Persistent overfitting despite frozen backbone: train_loss dropped 82% while val_loss increased 62%
- Val-test gap unchanged from parent (0.078 vs 0.080), revealing gap stems from focal loss class re-weighting ([3.0, 1.0, 5.0]) inflating val_f1 on small 167-sample validation set, not from LoRA
- Head over-parameterization (3-stage, 5.5M parameters) for frozen feature extraction
- Val-test performance anti-correlation from focal loss makes val_f1 unreliable for checkpoint selection
