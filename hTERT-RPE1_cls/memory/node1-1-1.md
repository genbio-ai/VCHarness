**Improvements from Parent**
- Fixed OOD input failure by replacing single-gene minimal profile with realistic multi-gene baseline (all 19,264 genes at expression=1.0, perturbed gene at 10.0)
- Upgraded backbone from AIDO.Cell-10M to AIDO.Cell-100M (102M params, 3.3M trainable via LoRA r=16)
- Removed class weights from focal cross-entropy (simpler loss: gamma=2.0 only)

**Results & Metrics (vs Parent)**
- Test F1: 0.3669 (+0.058 vs parent 0.3089), but still below MLP baseline (node1: 0.3762) and STRING_GNN (node4: 0.4258)
- Train-val loss gap: 0.0022 (vs parent 0.455) - near-zero gap indicating trivial convergence rather than healthy generalization
- Val F1 stability: std 0.0006 (vs parent 0.023) - noise-level variation confirming early saturation
- Val F1 progression: 0.3643 (epoch 0) → 0.3669 (epoch 21), plateauing after ~2 epochs

**Key Issues**
- **Mean-pool signal dilution**: averaging 19,264 gene embeddings reduces perturbed gene contribution to ~0.005%, preventing perturbation discrimination
- Trivial convergence: minimal train-val gap and near-zero val F1 variance indicate model lacks perturbation-discriminative capacity
- Performance gap: -0.0093 below MLP baseline, -0.0589 below STRING_GNN best performing node
