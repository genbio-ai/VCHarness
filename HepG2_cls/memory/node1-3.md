**Improvements from Parent**
- Replaced learned gene embedding table with pretrained STRING_GNN PPI embeddings (256-dim, frozen)
- Added ESM2-35M protein sequence embeddings (480→256 projected, frozen) via additive fusion
- Reduced model capacity from 8 residual blocks to 3 blocks (13.6M params vs 19.4M)
- Achieved +0.058 test F1 improvement (0.405→0.463)

**Results & Metrics (vs Parent)**
- Test F1: 0.463 (+0.058 vs parent 0.405, -0.011 vs tree best 0.474)
- Best val F1: 0.4625 at epoch 58
- Training epochs: 83 (early stopped)
- Overfitting gap: train loss 0.232 vs val loss 0.258 at best checkpoint
- Convergence: slow steady climb (val/F1 0.366→0.463 over 58 epochs) then plateau
- LR schedule: ReduceLROnPlateau (3e-4→9.4e-6 cascade, patience=5)

**Key Issues**
- ESM2 additive fusion consistently degrades performance across all tree nodes (every ESM2-augmented variant underperforms STRING-only baseline)
- ReduceLROnPlateau fires too early on noisy 141-sample validation set
- Output layer dominates parameters (10.2M of 13.6M total) creating overfitting bottleneck
- ESM2 protein sequence function not complementary to STRING PPI topology for HepG2 perturbation prediction
