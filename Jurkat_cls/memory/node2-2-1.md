**Improvements from Parent**
- Increased dual-pooling MLP head hidden dimension from 256 → 320
- Switched from cosine annealing to ReduceLROnPlateau scheduler

**Results & Metrics (vs Parent)**
- Test F1: 0.4472 vs 0.4453 (+0.002)
- Now best node in MCTS tree
- Training: 53 epochs vs 44 epochs (parent)
- Best val_f1: 0.447 at epoch 18 (parent: 0.445 at epoch 18)
- LR reductions: epochs 27/36/45 with 50% stepwise decay (3e-4 → 1.5e-4 → 7.5e-5 → 3.75e-5)
- Val_f1 trajectory: stable 0.438–0.444 band (eliminated parent's ±0.01 oscillation)

**Key Issues**
- Severe overfitting: val_loss 9.7× train_loss at epoch 53 (0.832 vs 0.086)
- Val_f1 never recovered above epoch 18 peak despite three LR reductions
- Architectural saturation at ~0.447 F1 ceiling with current approach
- Architecture saturated, not optimization-limited
