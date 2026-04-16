**Improvements from Parent**
- Adopted proven tree-best architecture from node2-1-3: partial STRING_GNN fine-tuning (frozen mps.0-5, trainable mps.6-7+post_mp, ~198K params) with 6-layer deep residual bilinear MLP head (rank=512, ~16.9M params)
- Reverted weight_decay from 1e-3 to 1e-4 ( abandoning parent's over-regularization strategy)
- Replaced parent's failed cond_emb injection and multi-layer attention fusion with focal cross-entropy loss (gamma=2.0, class weights [2.0, 0.5, 4.0])
- Used two-group AdamW: backbone_lr=5e-5, head_lr=5e-4

**Results & Metrics (vs Parent)**
- Test F1: 0.4704 vs parent 0.4024 (+0.068 improvement)
- Grandparent node4 F1: 0.4258 (+0.0446 improvement)
- Tree-best gap: 0.034 below node2-1-3 (0.5047)
- Training duration: 150 epochs, best at epoch 139 (val F1=0.4704)
- Val loss at best: 0.3802 vs train loss 0.0404 (9.4× ratio indicating severe overfitting)

**Key Issues**
- Cyclic cosine LR schedule: total_steps=1200 vs actual ~6600 steps causes LambdaLR cosine to wrap past progress=1.0 and cycle back to full LR every ~25 epochs, generating oscillatory multi-wave learning phases instead of monotonic decay
- Calibration overfitting: 16.9M-parameter head severely overfits 1,416 training samples, with dropout=0.2 insufficient (val/train loss ratio of 9.4× at best epoch)
- Performance gap from node2-1-3 (0.034 F1) attributed to LR schedule mismatch and head overfitting rather than architecture differences
