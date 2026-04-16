**Improvements from Parent**
- Added InductiveConditioningModule: 2-layer MLP (256→512→256, near-zero final init, ~262K params) adding learned residual offset to frozen PPI embeddings
- Added focal loss label smoothing (ε=0.05, γ=2.0)
- Increased early stopping patience to 50 epochs
- Calibrated cosine LR schedule (total_steps=4000, lr=5e-4)

**Results & Metrics (vs Parent)**
- Test F1: 0.4664 vs parent 0.4912 = −0.0248 (−5.1% regression)
- Best val_f1: 0.4662 at epoch 88 (val_loss=0.1298, train_loss=0.0592)
- Val/train ratio: 2.19× vs parent 3.71× (improved calibration)
- Trained 139 epochs with prolonged noisy plateau (epochs 20–60) and high epoch-to-epoch F1 variance (~0.007 std)
- Secondary improvement phase peaked at epoch 88, followed by high-variance post-peak drift (0.446–0.463)

**Key Issues**
- InductiveConditioningModule functions as unconditional PPI-topology-warping transformation, not truly perturbation-specific
- Learnable transformation introduces representational noise into proven frozen STRING_GNN embeddings
- Pattern of adding learnable transformations to frozen PPI embeddings failed consistently across 4 attempts (node1-2-1, node1-2-1-1, node1-3-1, node1-2-2)
- Label smoothing improved calibration but did not translate to F1 improvement
