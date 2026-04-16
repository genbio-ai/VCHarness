**Improvements from Parent**
- AIDO.Cell LoRA rank increased: r=16 → r=32 (higher capacity)
- STRING_GNN partially unfrozen: mps.6+mps.7+post_mp trainable at lr=1e-5 (frozen prefix as precomputed buffer)
- Fusion simplified: gated projection → simple concatenation+projection ([640∥256]→Linear(896→640)→GELU→LayerNorm)
- Regularization strengthened: head dropout=0.3, 3-group weight decay (backbone 0.01, GNN 0.01, head 0.05)
- Class weights softened: [10.91, 1.0, 29.62] → sqrt-inverse-frequency [3.3, 1.0, 5.44]

**Results & Metrics (vs Parent)**
- Test F1: 0.4683 vs 0.4797 (Δ -0.0114, regression)
- Val F1 peak: 0.4682 @ epoch 41 vs 0.4795 @ epoch 35 (Δ -0.0113)
- Val plateau: ~0.459±0.004 from epoch 41–81 (early stopping)
- Train loss: 0.025 (vs parent 0.044, better fit)
- Val loss: 0.841 (vs parent 3.004 at epoch 75, much better calibration)
- Train/val divergence ratio: 33.6× vs 68× (2× improvement, overfitting reduced)

**Key Issues**
- **Over-softened class weights**: up-regulated weight reduced from 29.62× to 5.44× (5.4-fold reduction)
- Insufficient gradient signal for rare up-regulated class (3% of labels), critical for per-gene macro-F1
- Parent's aggressive class weights were essential for minority class discrimination; calibration overfitting was symptom, not cause
