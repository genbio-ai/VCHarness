**Improvements from Parent**
- Replaced scFoundation frozen embeddings with AIDO.Cell-10M LoRA (r=16)
- Replaced GatedFusion (256+768→1024→256) with concat fusion MLP
- Added GenePriorBias with 20-epoch warmup (vs no warmup)
- Changed T_max to 120 (premature cosine decay)
- Reduced weight_decay to 3e-2 (lighter regularization)
- Reduced dropout to 0.35 (less regularization)
- Removed Mixup (alpha=0.2→none)

**Results & Metrics (vs Parent)**
- Test F1: 0.4647 vs parent 0.4669 (regression -0.002)
- vs sibling 0.4689 (regression -0.004)
- Best-in-tree AIDO.Cell-100M fusion: 0.5128 (gap -0.0481)
- Training trajectory: rapid improvement (val/f1 0.202→0.369, epochs 0–19), GenePriorBias disruption (epoch 20: 0.369→0.356), recovery (0.356→0.465 by epoch 83), 55-epoch plateau (epochs 83–138, val/f1 0.459–0.465), EarlyStopping at epoch 138
- Val-test gap ≈ 0 (no overfitting, converged to suboptimal local minimum)

**Key Issues**
- AIDO.Cell-10M fusion misalignment: 1-hot gene encoding → mean-pool pipeline produces fundamentally different signal than STRING_GNN's PPI topology
- Concat fusion MLP cannot cleanly merge orthogonal representations
- Bilinear head (5.1M params) amplifies representation confusion into degraded predictions
- Suboptimal hyperparameters: T_max=120 (proven: 200), weight_decay=3e-2 (proven: 4e-2), dropout=0.35 (proven: 0.5)
- 55-epoch plateau indicates convergence to poor local minimum
