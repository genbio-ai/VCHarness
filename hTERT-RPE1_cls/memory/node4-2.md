**Improvements from Parent**
- Partial backbone fine-tuning (mps.7+post_mp trainable, ~67K params; mps.0-6 frozen and cached) vs full fine-tuning (5.43M params)
- Deep 6-layer residual bilinear MLP head (hidden=512, expand=4, rank=512, dropout=0.3) vs simpler bilinear interaction head
- MuonWithAuxAdam optimizer (Muon lr=0.005 for 2D matrices, AdamW lr=1e-5/5e-4 for backbone/other) vs AdamW (lr=1e-4)
- Focal cross-entropy loss (gamma=2.0, class weights [2.0, 0.5, 4.0]) vs weighted cross-entropy (weights 12.28/1.12/33.33)
- Cosine warm restarts scheduler (T_0=600 steps ≈ 14 epochs/cycle) vs ReduceLROnPlateau

**Results & Metrics (vs Parent)**
- Test F1: 0.5069 vs parent 0.4258 (+0.081 improvement)
- Best val F1: 0.5069 at epoch 40 vs parent 0.4260 at epoch 32
- Training stopped at epoch 120 vs parent 53 epochs
- Staircase improvement across cycles 1-3 (cycle-best F1: 0.489→0.501→0.507)
- Within 0.003 of tree best (~0.510)

**Key Issues**
- Post-convergence overfitting after epoch 40: training loss decreased (0.0535→0.0281) while val F1 degraded (0.5069→0.4833) over epochs 40-120
- Small dataset (1,416 training samples) with large learnable gene output embedding (out_gene_emb: 6640×512 = 3.4M params) leads to memorization
- Early stopping triggered at epoch 120 (patience=80 from best at epoch 40)
