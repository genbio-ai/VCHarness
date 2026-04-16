**Technical Implementation**
- Model: AIDO.Cell-10M backbone with QKV-only fine-tuning
- Optimizer: Muon
- Feature extraction: Multi-layer fusion (last 4 transformer layers concatenated)
- Classification head: MLP with 10.7M parameters
- Loss: Label-smoothed cross-entropy with class weights
- Early stopping: patience=15, min_delta=1e-4
- Task: K562 single-gene DEG prediction
- Class imbalance: 92.5% neutral class

**Results & Metrics**
- Test F1: 0.426
- Parent (Node 1) F1: 0.370
- Improvement: +15.1%
- Training loss: 1.19 → 0.86
- Validation loss: 1.17 → 1.18
- Best val F1: 0.426 at epoch 13
- Final val F1: 0.403 at epoch 28
- Total epochs trained after peak: 16

**Key Issues**
- Clear overfitting pattern (training loss decreased while validation loss increased)
- Early stopping never triggered despite 16 epochs of post-peak degradation
- Insufficient regularization of large classification head
- LR scheduler settings allowed continued overfitting after peak performance
