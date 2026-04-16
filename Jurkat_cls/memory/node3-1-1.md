**Improvements from Parent**
- Replaced STRING GNN + ESM2 with AIDO.Cell-10M LoRA (r=4, last 3 layers, ~18K params)
- Introduced additive interaction head: pert_proj + gene_query → LayerNorm → Linear(256→3)
- Total trainable params: ~1.85M (1.70M gene_query embedding + 130K projection)
- Training: Focal Loss (γ=2, class weights, label_smoothing=0.05), differential LR (backbone=3e-5, head=3e-4), warmup+cosine schedule, weight_decay=0.01

**Results & Metrics (vs Parent)**
- Test F1: 0.2244 vs 0.2434 (parent node3-1) — regression of 0.019 F1
- Tree best: 47% below node1-1-1 (0.411 F1)
- Val_f1 oscillation: 0.023–0.224 across 45 epochs, ~40 direction changes
- Train loss: 4.07→3.50; Val loss: flat at ~4.03

**Key Issues**
- **Additive head architecture bottleneck**: addition provides rigid, position-invariant combination, cannot learn non-linear gene-perturbation interactions (proven concat+MLP heads achieve ≥0.40 F1)
- **Param allocation imbalance**: 93% of head params (1.70M gene_query) encode gene identity, only 771-param classifier handles interaction complexity
- **Severe underfitting**: val_f1 oscillation with no convergence, train-val gap confirms insufficient learning capacity
- **Architecture fundamentally flawed**: additive fusion proven inferior to concat+MLP across all successful AIDO.Cell nodes
