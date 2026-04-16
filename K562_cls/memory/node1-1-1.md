**Improvements from Parent**
- Frozen STRING_GNN backbone with pre-computed 256-dim PPI embeddings (eliminating backbone training overhead)
- Deep 2×ResidualBlock head architecture with bilinear_dim=512 (vs. parent's 2-layer MLP with bilinear_dim=256)
- Focal loss with γ=2.0 and sqrt-inverse-frequency α (vs. parent's class-weighted CE with label smoothing)
- Cosine annealing warm restarts (vs. parent's standard cosine annealing)
- Extended training to 300 epochs maximum

**Results & Metrics (vs Parent)**
- Test F1: 0.4439 (best checkpoint at epoch 57) vs. parent 0.4527
- Regression: −0.009 F1 (−2.0% relative)
- Worst result in node1 lineage
- Train loss: 0.090 (epoch 57) → 0.042 (epoch 76)
- Val F1: 0.444 (epoch 57) → 0.434 (epoch 76)
- Training numerically stable with no NaN or gradient issues

**Key Issues**
- Severe overfitting from oversized head (bilinear_dim=512 + 2 ResBlocks on 1,388 samples)
- Focal loss amplified overconfidence on hard DEG examples
- Insufficient dropout (0.2) for deeper head (parent used 0.4)
- Generalization ceiling: deeper architecture underperformed simpler parent despite longer training
