**Improvements from Parent**
- Expanded LoRA from last 6 layers to all 8 transformer layers
- Reduced LoRA rank from 8 to 4 (8.1M trainable params vs parent ~6M)
- Replaced dual pooling (gene positional + global mean-pool) with pre-computed STRING GNN PPI embeddings
- Simplified head from two-stage (640→256→19920) to single-stage (832→384→19920) MLP
- Added 3-branch character CNN (vs parent's multi-scale CNN)

**Results & Metrics (vs Parent)**
- Test F1: 0.4360 vs parent 0.4344 (+0.0016)
- Best val_f1: 0.5129 at epoch 13 vs parent 0.5142 at epoch 43 (-0.0013)
- Val loss at best val_f1: 0.370 vs parent 0.278 (+33%)
- Final val loss: 0.680 vs parent 0.463 (+47%)
- Val-test gap: 0.077 vs parent 0.080 (-0.003)
- Training stopped earlier (epoch 13 vs 43) due to faster overfitting

**Key Issues**
- Severe overfitting: val_loss increased 84% (0.370→0.680) while val_f1 remained stable
- Excessive LoRA capacity: ~5,400 params/sample (8.1M on 1,500 samples) enables memorization
- Val_f1 unreliable checkpoint metric: 167-sample val set's macro-F1 masks minority-class degradation
- Val_f1-val_loss anti-correlation from focal loss (gamma=2.0) amplifies confident wrong predictions
- Worst among node1's children: 0.006 below sibling frozen-backbone node1-1 (0.4420)
