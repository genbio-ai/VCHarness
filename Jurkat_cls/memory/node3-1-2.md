**Improvements from Parent**
- Replaced STRING GNN + ESM2-35M + per-gene MLP head with AIDO.Cell-10M LoRA (r=4 on all 8 layers)
- Added character-level Symbol CNN for additional feature representation
- Fused with frozen STRING GNN PPI embeddings as 4th source
- Changed to concat+MLP prediction head (832→384→19920, 11M params)
- Focal loss with class weights [5,1,10]

**Results & Metrics (vs Parent)**
- Test F1: 0.4577 vs parent 0.2434 (+0.214)
- Train loss: 0.726→0.354 (51% reduction)
- Val loss: 0.749→0.916 (22% increase, overfitting)
- Val F1 peaked at epoch 39, plateaued thereafter
- LR halved at epoch 48 with no further improvement

**Key Issues**
- Severe overfitting from epoch 3 onward (train loss ↓, val loss ↑)
- 11M parameter prediction head disproportionately large for 1,500 training samples
- Early stopping too loose: patience=18 with no min_delta, continued 18+ epochs past optimal
- Insufficient regularization (no weight decay, low dropout mentioned for next iteration)
