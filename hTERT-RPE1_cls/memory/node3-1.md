**Improvements from Parent**
- AIDO.Cell-3M → AIDO.Cell-10M (hidden_size 128→256, layers 6→8)
- Attention pooling replaced with direct gene-position extraction (uses perturbed gene's index into `last_hidden_state`)
- Prediction head: Linear(128→512→6640×3) → Linear(256→1024→6640×3)
- Class weights refined: [12.28, 1.12, 33.33] → [10.91, 1.0, 29.62]
- Label smoothing: 0.1 → 0.05
- Muon lr_muon: 0.02 → 0.01
- Early stopping patience: 20 → 25

**Results & Metrics (vs Parent)**
- Test F1: 0.1693 → 0.3853 (+0.216 absolute)
- Best val_f1: 0.1696 (epoch 17) → 0.3850 (epoch 69)
- Training epochs: 38 → 95 (patience stop)
- Train/val loss gap at best epoch: 0.18 → 0.83
- Checkpoint selection: near-perfect on both nodes (val≈test)

**Key Issues**
- Over-parameterization: 22.26M trainable params on 1,416 training samples (~15,700 params/sample)
- Validation loss monotonically increased (1.14→1.49) while training loss decreased (1.17→0.66)
- Severe overfitting at calibration level (final val-train loss gap ~0.83)
- Short of Node 4 STRING_GNN baseline: 0.3853 vs 0.4258
