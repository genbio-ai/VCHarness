**Improvements from Parent**
- Reduced model capacity: 4-block Pre-Norm MLP (h=512, inner=1024, dropout=0.40) → 3-block Pre-Norm MLP (h=384)
- Optimizer change: AdamW (lr=3e-4, wd=5e-4) → dual Muon+AdamW (Muon LR=0.01)
- Learning rate schedule: RLROP (patience=10, 3 halvings at epochs 70/164/175) → CosineAnnealingWarmRestarts (T_0=80, T_mult=2)
- Added Manifold Mixup data augmentation
- Removed per-gene bias (present in parent node3-3)
- Removed 5-epoch warmup phase

**Results & Metrics (vs Parent)**
- Test F1: 0.4636 vs 0.387 → +0.0766 improvement
- Improved over node3-3-1 (0.4226) by +0.0410
- Improved over node3-3-2 (0.4536) by +0.0100
- Below reference node1-3-3 (0.4950) by ~0.033
- Below reference node1-3-2-2-1-1-1-1-1-1-1-1 (0.4968) by ~0.033
- Below multi-modal ESM2+STRING ceiling (0.5175) by ~0.054
- Training epochs: 184 (early stopped 56 epochs before next scheduled warm restart at epoch 240)
- Converged to local minimum at epoch 123

**Key Issues**
- Muon optimizer stochastic trajectory instability: Newton-Schulz orthogonalization produces different convergence outcomes on different random initializations despite identical hyperparameters
- CosineAnnealingWarmRestarts cycle interrupted: early stopping patience too short to complete full restart cycle (T_0=80, stopped 56 epochs before next restart at epoch 240)
- Model trapped in local minimum after epoch 123 with warm restart at epoch 80 failing to provide escape
- STRING-only representation performance ceiling: ~0.463–0.497 versus multi-modal ESM2+STRING at 0.5175
- Manifold Mixup combination with Muon insufficient to overcome architecture ceiling
- 6,640 gene perturbation response classification task on HepG2 cells (per-gene 3-class)
