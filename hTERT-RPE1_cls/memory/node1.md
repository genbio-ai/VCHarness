**Technical Implementation**
- 4-layer residual MLP architecture (embed_dim=512, hidden_dim=1024)
- Low-rank output head with rank=512
- Random gene-symbol nn.Embedding (trained from scratch)
- 1,416 training samples, 6,640-dimensional DEG signature prediction
- Weighted cross-entropy loss with inverse-frequency class weights
- AdamW optimizer with ReduceLROnPlateau (patience=5, factor=0.5)
- Initial LR 3e-4, decayed to 4.6875e-6 over 81 epochs

**Results & Metrics**
- Test F1: 0.3762
- Best validation F1: 0.3763 at epoch 60
- Train loss: 0.82 → 0.17 (78.8% reduction)
- Validation loss: 0.77 → 1.90 (147.5% increase)
- Final train-to-val loss gap: 1.72
- Near-zero test-vs-validation generalization gap
- Hard plateau at F1≈0.375–0.376 in final 20 epochs
- 88.9% neutral class imbalance

**Key Issues**
- Severe and monotonic overfitting (train loss decreasing while validation loss increasing)
- Val F1 continued improving in Phase 2 despite loss divergence (F1 evaluates rank rather than calibration)
- Aggressive LR schedule causing premature convergence
- No biological prior knowledge in random embeddings with only 1,416 training samples
