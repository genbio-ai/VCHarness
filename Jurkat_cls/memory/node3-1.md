**Improvements from Parent**
- Replaced AIDO.Protein-16B LoRA protein sequence encoding with STRING GNN (5.43M params, 8 GCN layers) + ESM2-35M conditioning + per-gene MLP head
- Changed loss from standard CrossEntropy to Focal Loss (γ=2, class weights)
- Implemented AdamW with differential learning rates (GNN=1e-4, head=3e-4) + CosineAnnealing

**Results & Metrics (vs Parent)**
- Test F1: 0.2434 vs 0.0498 (4.9× improvement)
- Best val_f1: 0.2434 at epoch 28
- Final val_f1 (epoch 53): 0.2278
- Train loss: 3.94→3.04
- Val loss: ~3.98 (flat throughout training)
- Train-val gap: widened from 0.25→0.98 (severe underfitting/memorization)
- Val_f1 oscillated chaotically: 34 direction changes across 52 epochs

**Key Issues**
- GNN forward pass is sample-invariant: ESM2 conditioning computed once per batch from static pre-computed embeddings, producing identical [18870, 256] outputs for 93.5% of genes
- Only gene_query embedding (1.70M params) provides per-sample differentiation—an impossible burden
- Static GNN output architecture prevents per-sample differentiation; requires query-based GNN node retrieval, per-sample expression encoding (e.g., AIDO.Cell), or abandoning PPI graph approach
