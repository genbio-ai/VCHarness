**Improvements from Parent**
- Switched from AIDO.Cell-100M to AIDO.Cell-10M with reduced LoRA rank (r=8 vs r=16, applied to last 4 of 8 layers instead of layers 6–17 on Q/K/V)
- Added character-level CNN gene symbol encoder (3-branch Conv1d → 64-dim) to capture gene family naming conventions (NDUF, KDM, DHX prefixes)
- Changed head architecture to dual-pooling MLP (576→256→19920)
- Replaced ReduceLROnPlateau with cosine annealing learning rate schedule

**Results & Metrics (vs Parent)**
- Test F1: 0.4453 vs 0.404 (+0.041 improvement)
- Best val_f1: 0.445 at epoch 18 vs 0.404 at epoch 51
- Trained 44 epochs with early stopping (vs 67 epochs)
- New best node in MCTS tree, also +0.035 above sibling node2-1 (0.410)
- Severe overfitting: train_loss decreased 0.413→0.110 while val_loss increased 0.456→0.836
- Val_f1 oscillated between 0.428–0.445 without surpassing epoch 18 peak

**Key Issues**
- Severe overfitting with widening train-val loss gap
- Persistent val_f1 oscillation driven by cosine annealing schedule causing val_loss spikes up to 0.897
- Optimization instability from high learning rates and cosine decay preventing val_f1 recovery
