**Technical Implementation**
- AIDO.Cell-100M with LoRA (r=16) fine-tuning
- Focal loss (γ=2.0)
- 512-dim fully fine-tuned head (10.1M params)
- Mean-pool over 19,264 gene positions
- ReduceLROnPlateau scheduler
- EarlyStopping with min_delta=1e-4

**Results & Metrics**
- Test F1=0.4425
- 71 epochs trained
- Best val F1 at epoch 55
- Final val F1 degraded to 0.432
- Train loss collapsed to 0.0036
- +19.6% over parent node1 (0.3700)
- -0.0102 vs sibling node1-1 (0.4527)

**Key Issues**
- Severe overfitting: train loss 0.0036 with val F1 degradation
- EarlyStopping bug: min_delta=1e-4 too tight for val F1 oscillation ±0.005
- Domain mismatch: AIDO.Cell encodes steady-state co-expression, DEG requires perturbation propagation
- Mean-pool dilutes signal with 99.99% missing values (-1.0)
- ReduceLROnPlateau second LR reduction at epoch 64, 9 epochs post-peak
