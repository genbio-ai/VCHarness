**Improvements from Parent**
- Partial STRING_GNN fine-tuning (mps.0-6 frozen, mps.7+post_mp trainable at lr=1e-5) vs full fine-tuning
- 6-layer residual bilinear MLP head (hidden=512, expand=4, rank-512 bilinear) vs simple bilinear head
- MuonWithAuxAdam optimizer (Muon lr=0.005 for 2D matrices, AdamW lr=5e-4 for head) vs AdamW (lr=1e-4)
- Focal cross-entropy loss (gamma=2.0, class_weights=[2.0, 0.5, 4.0]) vs weighted cross-entropy (weights 12.28/1.12/33.33)
- Cosine warm restarts scheduler (T_0=1200) vs ReduceLROnPlateau
- Three regularization measures vs node4-2: dropout 0.3→0.4, out_gene_emb weight_decay 1e-3→1e-2, gradient_clip_val=1.0

**Results & Metrics (vs Parent)**
- Test F1: 0.5036 (parent node4: 0.4258, sibling node4-2: 0.5069)
- Best validation F1: 0.5034 at epoch 73 (parent: 0.4260 at epoch 32)
- Early stopping: epoch 123, patience=50 (parent: epoch 53, patience=20)
- Training dynamics: W-shaped cycle pattern (peaks: 0.5030→0.4986→0.5034→0.4966→0.4874) with high intra-cycle variance
- Overfitting gap at best epoch: train_loss=0.0378 vs val_loss=0.3532 (gap=0.3154)

**Key Issues**
- Over-regularization: simultaneous application of three stronger regularization measures (dropout 0.4, out_gene_emb wd 1e-2, gradient clipping) suppresses peak representational capacity below node4-2's ceiling
- Cycle instability: longer T_0=1200 step cycles introduce instability at cycle boundaries, unlike node4-2's cleaner staircase convergence
- Severe overfitting persists: large train-val loss gap (0.3154) indicates model memorizes training data despite regularization
- Performance regression: test F1 0.0033 below sibling node4-2, suggesting regularization balance is suboptimal
