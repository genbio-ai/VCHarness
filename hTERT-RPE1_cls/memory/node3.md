**Technical Implementation**
- Model: AIDO.Cell-3M (hidden_size=128, 6 transformer layers)
- Fine-tuning: QKV-only direct fine-tuning
- Optimizer: Muon (lr_muon=0.02, lr_adam=3e-4) with ReduceLROnPlateau scheduler
- Pooling: Learnable single-query attention pooling over all 19,264 gene positions → 128-dim vector
- Prediction head: Linear(128→512→6640×3)
- Loss: Cross-entropy with class weights [12.28, 1.12, 33.33], label_smoothing=0.1
- Training: 38 epochs with early stopping (patience=20)

**Results & Metrics**
- Best val_f1: 0.1696 at epoch 17
- Test F1: 0.1693 (near-perfect checkpoint selection, no val-to-test gap)
- 8-epoch warm-up phase: val_f1 stagnated at ~0.063
- Three LR reductions: 0.02→0.01→0.005→0.0025
- Validation loss: monotonically increased from 1.40 to 1.50

**Key Issues**
- Severe representation bottleneck: collapsing 19,264 gene states into single 128-dim vector
- Single-query attention pooling discards per-gene positional information
- AIDO.Cell-3M minimal hidden size insufficient for 6,640×3 structured prediction
- Loss-level overfitting: validation loss rose throughout training despite F1 peak
- Non-monotonic dynamics with irreversible decay after epoch 17
