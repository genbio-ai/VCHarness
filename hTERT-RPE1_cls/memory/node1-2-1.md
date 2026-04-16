**Improvements from Parent**
- Partial fine-tuning of STRING_GNN: last 2 GCN layers + post_mp (~530K additional trainable params)
- Per-sample perturbation conditioning via batch-summed `cond_emb` injection into GNN forward pass
- STRING_GNN-initialized output gene embeddings [6640, 256] from first N_GENES_OUT GNN nodes
- Two-group AdamW optimizer: backbone lr=5e-5, head lr=5e-4 (single lr=5e-4 in parent)

**Results & Metrics (vs Parent)**
- Test F1: 0.4500 vs 0.4912 (-0.0412, -8.4% regression)
- Peak val F1: 0.450 vs 0.4911
- Initial epoch-0 val F1: 0.290 vs 0.351
- Training epochs: 129 (same as parent), early stopping at epoch 98 + patience 30 (same trigger point)
- Val/train loss ratio: 1.70× vs 3.71× (improved calibration but worse discriminability)

**Key Issues**
- Batch-level `cond_emb` mixing flaw: sums perturbation signals from all 16 batch samples simultaneously into single GNN forward pass
- 8 GCN layers with 8 hops of message passing cause signals from different perturbed genes to mix across shared PPI neighborhoods
- Batch-composition-dependent corrupted embeddings destroy perturbation-specificity
- Misaligned biological priors in output gene initialization: first 6,640 GNN nodes ≠ actual 6,640 label positions
