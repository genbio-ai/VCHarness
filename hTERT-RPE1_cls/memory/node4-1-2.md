**Improvements from Parent**
- Partial fine-tuning of STRING_GNN backbone: only `mps.7 + post_mp` trainable (~67K params) with frozen precomputed cache for layers mps.0–6
- Deep residual bilinear head: 6-layer rank-512 MLP (~16.9M params) replacing parent's 2-layer interaction head
- MuonWithAuxAdam optimizer: Muon lr=0.005 for 2D ResBlock matrices, AdamW lr=5e-4 for head scalars, AdamW lr=1e-5 for backbone
- CosineAnnealingWarmRestarts scheduler (T_0=600 steps, T_mult=1) replacing parent's CosineAnnealingLR
- Weight decay reverted to parent-level (implicit, addressing node4-1's 1e-3 over-regularization)

**Results & Metrics (vs Parent)**
- Test F1: 0.5060 vs 0.4024 (+0.1036 improvement)
- Validation F1 peak: 0.5059 at epoch 10 (vs parent: 0.4027 at epoch 166)
- Training epochs: 91 (vs parent: 191)
- Training loss: 0.255 → 0.019 (13.5× reduction; vs parent: stalled at ~0.89)
- Near MCTS tree best: 0.5060 vs 0.5099 (gap: -0.0039)
- Substantially outperformed sibling node4-1-1: 0.5060 vs 0.4704

**Key Issues**
- Severe overfitting: train_loss collapsed to 0.019 while val_F1 declined from 0.5059 to 0.4752
- Anti-staircase behavior across warm restart cycles: val_F1 peaks 0.5059 → 0.4938 → 0.4890 → 0.4843
- T_0 miscalibration: 600 steps ≈ 26 epochs/cycle (not 14 as intended), only ~3.5 cycles in 91 epochs
- Extreme parameter-to-sample ratio: ~11,900 head parameters per training sample (1,416 samples)
- Head overfitting bottleneck: 6-layer residual MLP rapidly memorizes training data, preventing warm restarts from discovering new peaks beyond cycle 0
