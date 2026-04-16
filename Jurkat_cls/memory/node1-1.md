**Improvements from Parent**
- Replaced character-level BiGRU with frozen AIDO.Cell-100M pretrained backbone
- Implemented global mean-pool embedding strategy for synthetic expression profiles
- Switched optimizer to AdamW with lr=1e-3 and CosineAnnealingLR scheduler
- Maintained focal loss (γ=2) and class weighting from parent

**Results & Metrics (vs Parent)**
- Test F1: 0.390 (identical to parent: 0.390)
- Best val_f1: 0.469 at epoch 211 (worse than parent: 0.509 at epoch 19)
- Training loss: 2.89→1.89 (decreased)
- Val_loss: 3.30→4.06 (increased, indicating severe overfitting)
- Performance parity with character-level approach despite pretrained embeddings
- Significantly below node2's LoRA fine-tuning (val_f1=0.783)

**Key Issues**
- Synthetic expression profile strategy (perturbed gene=10.0, all others=-1.0 mean-pooled) produces near-uninformative one-hot-like embeddings
- Frozen backbone contributes zero biological signal — embeddings indistinguishable across samples
- MLP head memorized 1,500 training embeddings with no generalization
- Training loss decreased while val_loss increased simultaneously — classic memorization pattern
- Frozen embedding approach is a dead end for this task
- Required 211 epochs vs parent's 19 epochs with worse results
