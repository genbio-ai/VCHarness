**Improvements from Parent**
- Focal loss with gamma=2.0 replacing label-smoothed cross-entropy
- Learnable attention-weighted layer fusion (256-dim weighted sum) replacing 4-layer concatenation

**Results & Metrics (vs Parent)**
- Test F1: 0.188 (vs parent 0.426, -56% relative drop)
- Val F1: 0.414 at epoch 24
- Prediction distribution: ~33% per class (collapsed from expected 92.5% neutral)

**Key Issues**
- Focal loss + class weights combination caused extreme gradient amplification for minority classes
- Model collapsed into near-uniform class predictions
- Neutral class predictions destroyed, essential for high F1 in this imbalanced task
