**Improvements from Parent**
- Switched from AIDO.Cell-100M to AIDO.Cell-10M
- LoRA r=8 on last 4 layers instead of r=16 on layers 6-17
- Added character-level gene symbol CNN encoder
- ReduceLROnPlateau scheduler (patience=8, factor=0.7)
- 320-dim prediction head (vs parent dual-pooling head)
- Focal loss gamma=1.5 (vs parent gamma=2)
- weight_decay=0.03 (vs parent no explicit decay)
- Checkpoint averaging implementation

**Results & Metrics (vs Parent)**
- Test F1=0.4375 (+3.35% vs parent 0.404)
- Best val_f1=0.437 at epoch 27
- Early improvement: val_f1 0.367→0.408 (epochs 0-5)
- Peak plateau: val_f1=0.433 at epoch 16
- Training: 62 epochs total, early stopping triggered
- Overfitting pattern: train_loss 0.276→0.155, val_loss 0.581→0.646 (final 35 epochs)

**Key Issues**
- Insufficient regularization for 1,500 samples (weight_decay=0.03, focal_gamma=1.5, 320-dim head)
- ReduceLROnPlateau triggered too late (after overfitting began)
- Test F1=0.4375 below sibling node2-2 (0.4453) and 0.447 target
- Top-3 checkpoints (epochs 16/20/27) clustered with minimal diversity
- AIDO.Cell-10M + symbol CNN ceiling ~0.447
