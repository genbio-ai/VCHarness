**Improvements from Parent**
- Multi-pooling (mean+max, 832-dim input) replacing single mean-pooling
- Expression noise augmentation (std=0.02)
- CosineAnnealingLR replacing ReduceLROnPlateau
- Restored regularization: weight_decay=0.05, focal_gamma=2.0, head_hidden=256

**Results & Metrics (vs Parent)**
- Test F1=0.4358 vs parent 0.4375 (Δ=-0.0017, regression)
- Lineage-best comparison: 0.4472 (node2-2-1, Δ=-0.0095)
- Training duration: 59 epochs
- Best val_f1=0.4325 at epoch 39
- Post-peak degradation: 20 epochs with val_f1 decline and val_loss increase (0.455→0.496)

**Key Issues**
- Multi-pooling failed: mean+max of synthetic one-hot embeddings adds positional noise, not complementary biological signal
- Expression noise augmentation degraded clean binary input encoding
- CosineAnnevingLR caused oscillation without recovery above epoch 39
- Persistent overfitting: 20 epochs of val_f1 degradation with val_loss increase after epoch 39
- Synthetic one-hot input paradigm hitting ceiling ~0.447 across 5+ generations of node2 lineage
