**Improvements from Parent**
- Removed ESM2 protein sequence embeddings (reverted to STRING-only)
- Added factorized output head: 512→256→19920 bottleneck with GELU, dropout=0.2
- Per-gene bias term in output layer

**Results & Metrics (vs Parent)**
- Test F1: 0.430 (parent: 0.463, delta: −0.033)
- Best val F1: 0.4407 at epoch 25 (parent: 0.4625 at epoch 58)
- Training duration: 51 epochs (early stopped after 2 LR halvings)
- Train-val loss gap: ~0.06 (mild overfitting)
- Tree ceiling: 0.474 (node1-1-1/node4), gap: −0.044

**Key Issues**
- Factorized output head (512→256→19920) causes significant capacity loss — third independent failure of 256-dim bottleneck (node1-1-2: 0.436, node3-2-1-1: 0.358, this node: 0.430)
- 256-dim bottleneck destroys fitting capacity on 19,920-output classification without providing regularization benefit
- Performance regression confirms additive ESM2 fusion was not the primary bottleneck in parent node1-3
