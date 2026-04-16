**Improvements from Parent**
- Increased fusion output dimension from 256 to 512 (ExpandedGatedFusion)
- Added per-modality LayerNorm before GatedFusion gate
- Added residual bypass connection from STRING_GNN embeddings (fused at output: gate_output + residual)
- Removed Mixup (previously alpha=0.2)
- Changed learning rate schedule to cosine annealing (T_max=100)
- Implemented discriminative layer-wise LR: 3e-4 for fusion module, 1e-4 for bilinear head (3× ratio)

**Results & Metrics (vs Parent)**
- Test F1: 0.4689 (+0.002 vs parent node1-3: 0.4669)
- Val F1: 0.4689 (zero val-test gap)
- Training: 100 epochs completed (EarlyStopping never triggered)
- Best checkpoint: epoch 84
- 48 incremental improvements over 84 epochs
- Train-val loss gap: 0.228 (capacity-limited convergence, not overfitting)
- Still below STRING_GNN-only baseline (node1-2: 0.4769, delta -0.008)
- Still below best fusion (node4-2: 0.4801, delta -0.011)

**Key Issues**
- Frozen scFoundation embeddings encode steady-state transcriptomic patterns rather than perturbation-specific responses
- Fusion architecture fundamentally misaligned with DEG prediction task
- Residual bypass provides only partial floor but cannot eliminate fusion noise from GatedFusion gate processing irrelevant scFoundation signal
