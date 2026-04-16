**Improvements from Parent**
- Reduced MLP head from 5 residual blocks to 3 residual blocks
- Partially unfroze STRING_GNN pretrained backbone (last 2 layers at LR=5e-5)
- Added per-gene bias layer with 19,920 parameters
- Increased dropout from 0.3 to 0.35
- Replaced cosine annealing LR schedule with ReduceLROnPlateau

**Results & Metrics (vs Parent)**
- Test F1: 0.474 vs parent 0.472 (+0.002, +0.4%)
- Training duration: 74 epochs vs parent 53 epochs
- LR reductions: two steps at epochs 52-53 (3e-4→1.5e-4) and epoch 61 (7.5e-5)
- Val F1 temporarily recovered from 0.459 to 0.471 after LR reductions
- Training loss: 0.081→0.012 (6.75× reduction)
- Validation loss: 0.040→0.058 (1.45× increase)
- Trainable parameters: ~13.5M

**Key Issues**
- Overfitting persists with same train/val loss divergence pattern as parent
- Training loss decreased while validation loss increased simultaneously
- Output head dominates model capacity: 10.2M parameters (512→19,920)
- Model capacity (13.5M params) far exceeds training sample size (1,273 samples)
