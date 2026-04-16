**Improvements from Parent**
- Switched from AIDO.Cell-100M to AIDO.Cell-10M to reduce model capacity
- Reduced LoRA rank from r=16 to r=8, applied to all 8 layers (vs layers 6-17 on Q/K/V in parent)
- Added LoRA dropout=0.25 for regularization
- Changed class weights to [5.0, 1.0, 10.0]
- Increased weight_decay to 1e-2
- Replaced ReduceLROnPlateau with OneCycleLR scheduler

**Results & Metrics (vs Parent)**
- Test F1: 0.4101 vs parent 0.404 (+0.006)
- Best val_f1: 0.4101 at epoch 31 (vs parent 0.404 at epoch 51)
- Training epochs: 51 (vs parent 67)
- Train loss: 0.406 → 0.038
- Val loss: 0.430 → 1.03 (increased throughout training)
- Val_f1 plateaued at 0.40–0.41 from epoch 7 onward

**Key Issues**
- Persistent overfitting despite smaller model and stronger regularization
- Synthetic one-hot input (perturbed_gene=0, others=1.0) provides only positional information, no biologically meaningful co-expression signal
- Task equivalent to lookup table memorization
- Regularization reduces overfitting magnitude but does not improve val_f1 ceiling
