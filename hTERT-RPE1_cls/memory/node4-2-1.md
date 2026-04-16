**Improvements from Parent**
- Extended cosine warm restart cycle from T_0=600 to T_0=1200 steps (~27 epochs/cycle)
- Increased dropout from 0.3 to 0.4 in the 6-layer residual bilinear MLP head
- Added stronger weight decay (1e-2) on the out_gene_emb parameter group (3.4M params)

**Results & Metrics (vs Parent)**
- Test F1: 0.5076 (parent: 0.5069) → +0.0007 improvement
- Best val F1: 0.5073 at epoch 52 (parent: 0.5069 at epoch 40)
- Early stopping at epoch 102, patience=50 (parent: epoch 120, patience=80)
- Training cycle pattern: Cycle 1 peaked at 0.5014 (epoch 25), Cycle 2 peaked at 0.5073 (epoch 52, global peak), Cycles 3-4 degraded (0.505→0.490)
- Train loss continued declining post-peak: 0.0488→0.0338 over 50 epochs

**Key Issues**
- Fundamental parameter-to-sample imbalance: 17M trainable parameters vs 1,416 samples (~12,000:1 ratio)
- 3.4M-param out_gene_emb matrix dominates parameter count
- Stronger regularization (dropout 0.4, gene_emb WD 1e-2) reduced but did not eliminate post-peak overfitting
