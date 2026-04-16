**Improvements from Parent**
- Reverted focal loss (gamma=2.0) to label-smoothed cross-entropy + class weights
- Extended QKV-only fine-tuning to QKV+Output (adding attention.output.dense to Muon-optimized params)
- Added CosineAnnealingWarmRestarts LR schedule (T_0=50)

**Results & Metrics (vs Parent)**
- Test F1: 0.4325 vs 0.188 (+0.2445 absolute, +130% relative)
- Val F1: 0.431 best at epoch 15; final val F1=0.431 at epoch 26 (no early stop)
- Train-val loss gap: ~0.18 (mild overfitting)
- QKV+Output vs QKV-only: +0.006 F1 (+1.4% relative over grandparent node3's 0.426)
- Model params: 10.7M classification head on 1,388 samples

**Key Issues**
- Large classification head (10.7M params on 1,388 samples) drives persistent overfitting
- CosineAnnealingWarmRestarts functionally equivalent to standard cosine decay (only 26/50 epochs used)
- AIDO.Cell-10M provides static gene expression embeddings, not perturbation-aware representations
- F1 ceiling ~0.43 below STRING_GNN baseline (node1-1: 0.453) due to fundamental embedding limitation
