**Improvements from Parent**
- Reduced MLP head dimension from 384 to 256
- Increased dropout to 0.5
- Increased weight decay to 0.05
- Increased label smoothing to 0.1
- Changed PPI projection from 1-layer to 2-layer
- Fixed ReduceLROnPlateau to monitor val_loss instead of val_f1

**Results & Metrics (vs Parent)**
- Test F1: 0.4419 (parent: 0.462, regression: -0.020)
- Val loss at epoch 1: 0.530 (parent: 0.426 at E1)
- Best val F1: 0.4419 at epoch 21
- LR scheduler fired twice at epochs 10 and 16
- Training epochs: 40

**Key Issues**
- Severe underfitting: val_loss catastrophically high from epoch 0
- Capacity-starved model: 256-dim head reduced params from ~3.2M to ~1.7M for 6,640-gene output with 1,500 training samples
- Excessive combined regularization: dropout 0.5 + weight decay 0.05 + label smoothing 0.1 compounded underfitting
- LR scheduler fired correctly but could not rescue underfit model
