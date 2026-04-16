**Improvements from Parent**
- Added PPI Neighborhood Attention Module (K=16 neighbors, attn_dim=64, +230K parameters) to aggregate PPI network context
- Extended training budget: max_epochs=300 (vs 200), patience=35 (vs 25)
- Reduced warmup duration to 5 epochs (vs 10)
- Lowered LR floor: min_lr_ratio=0.08 (vs 0.15)
- Increased Mixup strength: alpha=0.4 (vs 0.2)

**Results & Metrics (vs Parent)**
- Test F1: 0.4801 (vs 0.4801, no improvement)
- Best validation F1: 0.4836 at epoch 134 (vs 0.4801 at epoch 139, +0.0035)
- Training duration: 170 epochs executed
- Loss at best checkpoint: train=0.924, val=1.079 (train<val anomaly)
- Val-test gap: 0.0035 (vs ~0.000 for parent, +0.0035 degradation)

**Key Issues**
- **Neighborhood attention redundancy**: PPI neighbor context overlaps with scFoundation's perturbation-aware fusion signal, failing to provide complementary information (technique provided +0.010 F1 in STRING_GNN-only lineage but did not transfer)
- **Overfitting validation set**: +230K parameters overfit 154-sample validation set, creating 0.0035 val-test gap where parent had near-zero generalization gap
- **Train-val loss inversion**: Stronger Mixup (alpha=0.4) artificially suppresses training loss, creating anomalous train<val pattern (0.924 vs 1.079)
- **No test set transfer**: Validation F1 improved by 0.0035 at best checkpoint but test F1 remained identical to parent
