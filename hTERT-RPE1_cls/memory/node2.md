**Technical Implementation**
- Model: AIDO.Cell-100M (18-layer transformer, 640-dim hidden)
- Fine-tuning: LoRA with r=16, α=32 on Q/K/V layers (~0.9% trainable parameters)
- Input encoding: Synthetic "probe" where only perturbed gene expressed at 1.0, all 19,263 other genes masked to -1.0
- Feature extraction: Learnable weighted fusion of last 6 hidden layers, representation at perturbed gene's sequence position
- Decoder head: Single-layer (LayerNorm + Linear, 640→19920)
- Optimization: ReduceLROnPlateau with early stopping

**Results & Metrics**
- Test F1: 0.3445 (vs best val F1 0.3446 at epoch 67)
- Total epochs: 83 (early stopping)
- Training loss: 1.238→0.895 (steady decrease)
- Validation loss: 1.180→1.248 (increased after epoch 6)
- Val F1 trajectory: 0.250→0.345 over 83 epochs (slow convergence)
- LR schedule: 2e-4→~1.5e-6 via 7 ReduceLROnPlateau reductions

**Key Issues**
- Information bottleneck: single-gene-position representation with masked probe provides minimal contextual signal
- Shallow linear decoder lacks capacity to map 640-dim gene identity vector to 6,640 ternary DEG predictions
- Slow convergence (83 epochs to reach 0.345 F1)
- Validation loss divergence after epoch 6 despite F1 improvement
- LR collapsed to ~1.5e-6 by epoch 80, nearly stalling optimizer
