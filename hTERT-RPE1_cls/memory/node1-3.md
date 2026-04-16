**Improvements from Parent**
- Partial STRING_GNN backbone fine-tuning: last 2 GCN layers + post_mp (~530K trainable params)
- Post-GNN low-rank additive perturbation conditioning: rank-16 factored pert_matrix (~307K params, residual connection without batch-mixing flaw)
- STRING_GNN-initialized output gene embeddings for two-sided bilinear interaction
- 6-layer residual MLP head with hidden_dim=512, dropout=0.3
- Two-group AdamW optimization: backbone lr=5e-5, head lr=5e-4, weight_decay=1e-3
- Properly-calibrated cosine annealing scheduler (total_steps=1,650)
- Focal loss with γ=2.0

**Results & Metrics (vs Parent)**
- Test F1: 0.4120 (parent node1-2: 0.4912) — **regression of −0.0792 (−16.1%)**
- Best val F1: 0.4121 at epoch 87 (of 138 total)
- Test-val gap: +0.0001 (correct checkpoint selection)
- Training dynamics healthy: final epoch loss gap 0.038 (val 0.114 vs train 0.076)
- Val F1 secondary improvement window: epochs 60–87 added +0.012 F1
- Lower initial epoch-0 val F1: 0.2895 vs. parent node1-2's 0.3510

**Key Issues**
- Partial backbone fine-tuning destabilized pre-trained PPI representations: 530K backbone params received insufficient gradient signal from 1,416 samples, disrupting structurally-optimized topology
- Semantically misaligned out_gene_emb initialization: first 6,640 STRING_GNN nodes arbitrarily mapped to DEG label positions (node ordering not aligned to label positions), creating biased prior that misleads bilinear optimization
