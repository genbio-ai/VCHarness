**Improvements from Parent**
- Switched from AIDO.Cell-100M with LoRA to partially fine-tuned STRING_GNN backbone (frozen mps.0–5 pre-computed as buffer; trainable mps.6, mps.7, post_mp; ~198K trainable backbone params)
- Replaced large MLP head with rank=512 6-layer residual bilinear MLP head (~16.9M params)
- Added strong class weighting to focal loss: [down=2.0, neutral=0.5, up=4.0]
- Implemented two LR groups with 10× separation (backbone=5e-5, head=5e-4)
- Set cosine annealing with total_steps=6600, warmup=100, patience=50

**Results & Metrics (vs Parent)**
- Test F1: 0.5047 vs 0.42337 (+0.0813, ~19.2% relative improvement)
- Achieved new tree-best F1, surpassing sibling node2-1-2 (0.5011) by +0.0036
- Best val F1=0.5047 at epoch 32
- Training loss: 0.080→0.041 over epochs 32–82
- Validation loss: 0.210→0.299 over same period (overfitting after peak)
- Training stopped at epoch 83 due to patience

**Key Issues**
- LR schedule miscalibration: total_steps=6600 designed for ~600 epochs, but training stopped at epoch 83, causing cosine LR to remain essentially flat (5e-4→4.82e-4)
- Moderate-to-severe overfitting after epoch 32: val loss increased 42% (0.210→0.299) while train loss continued declining
- Insufficient regularization for high-capacity head (16.9M params on 1,416 samples): dropout=0.2, weight_decay=1e-3 inadequate
- No secondary improvement phase (unlike node2-1-2's slow post-peak gains)
