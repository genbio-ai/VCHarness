**Improvements from Parent**
- Architecture: Changed from QKV(all 8 layers) + FFN(last 4 layers) to QKV+FFN(2L) trainable
- Optimization: Changed from cosine LR with 3-epoch warmup to SGDR T_0=12 schedule
- Architecture: Added GenePriorBias ([3,6640] additive head bias)
- Architecture: Reduced head_hidden from default to 384
- Regularization: Added 2% DEG label noise

**Results & Metrics (vs Parent)**
- Test F1: 0.4134 vs parent 0.4175 (Δ = -0.0041)
- Test F1 vs best sibling node3-3-2: 0.4496 (Δ = -0.0362)
- Peak val F1: 0.413 at epoch 32 (parent: 0.4175 at epoch 25)
- Training duration: 61 epochs, early stopping
- Final train loss: 0.858

**Key Issues**
- GenePriorBias (19,920 params) learned unstable priors with limited training data (1,388 samples)
- Reduced head capacity (head_hidden=384) insufficient for perturbation-specific patterns
- SGDR T_0=12 cycle 1 (24 epochs) too short for proper post-restart recovery
- Severe overfitting: train loss decreased steadily while val F1 plateaued at epoch 32
- Erratic oscillation throughout training due to aggressive SGDR cycling
