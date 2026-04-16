**Improvements from Parent**
- Changed LR schedule from cosine annealing to SGDR (T_0=18, T_mult=2, warmup=5)
- Added 3% DEG label noise

**Results & Metrics (vs Parent)**
- Test F1: 0.4296 (+0.0001 vs parent's 0.4295, -4.5% vs node3-3-2's 0.4496)
- SGDR restart at epoch 22 caused severe overfitting: val F1 peaked at 0.430 (epoch 24) then collapsed to 0.392 (epoch 30) — 0.038 drop
- Training loss dropped from 1.003 to 0.895 during collapse, confirming memorization
- Parent's cosine schedule maintained stable ~0.42 val F1 through same window

**Key Issues**
- SGDR restart with full-peak LR (0.02 vs parent's 73% peak of 0.0147) amplifies overfitting of 10.7M-parameter head on 1,388 samples
- 3% label noise combined with SGDR restart creates doubly-destabilizing dynamics, corrupting sparse DEG signal
- Cosine schedule never entered meaningful decay phase (warmup consumed 16% of training, early stopped before LR could fully decay)
