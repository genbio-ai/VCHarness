**Improvements from Parent**
- FFN (SwiGLU) unfreezing in last 2 transformer layers (~2.56M backbone trainable params)
- Increased regularization: dropout 0.2→0.35, weight_decay 1e-2→2e-2
- Tightened early stopping: patience 15→6, min_delta 1e-4→0.003
- Improved LR schedule: warmup 10→5 epochs, min_ratio 0.0→0.05 (cosine annealing)
- Adjusted base LR: 0.02→0.025

**Results & Metrics (vs Parent)**
- Test F1: 0.4295 vs 0.426 (+0.7% improvement)
- Val F1: 0.4294 vs 0.426 (+0.8% improvement)
- Val-test gap: ~0.0001 (near-zero) vs parent's train-val gap of 0.31
- Best val F1 at epoch 24 vs epoch 13
- Early stopping triggered at epoch 31
- Val F1 trajectory: 0.191 (epoch 0) → 0.429 (epoch 24)
- Overfitting eliminated: val loss stable at 1.17 vs parent's 1.17→1.18 increase

**Key Issues**
- Cosine LR schedule barely entered decay phase (warmup consumed 16% of training, peak LR reached 73% of base before early stopping)
- Optimizer never explored higher LR values or benefited from scheduled decay to finer minima
- Max_epochs=45 insufficient for full cosine annealing cycle
- Warmup period too long relative to total training duration
