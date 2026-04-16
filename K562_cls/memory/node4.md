**Technical Implementation**
- Heterogeneous multi-modal fusion: scFoundation (top-4 layers partial FT, 28M params) + STRING_GNN (full FT, 5.43M params)
- Gated cross-attention fusion module for 3-class differential expression prediction (6,640 genes per perturbation)
- Focal loss with class weights
- ~33M trainable parameters on 1,388 training samples
- ReduceLROnPlateau learning rate schedule

**Results & Metrics**
- Test F1: 0.4585 (best among all completed nodes)
- Outperformed node1-1 (0.453) and node3 (0.426)
- Training loss: 0.005–0.009; Validation loss: 0.091 (18× gap)
- Validation F1 plateaued at epoch 66, then fluctuated
- Near-zero train/test generalization gap

**Key Issues**
- Severe overfitting: 18× validation/training loss gap (0.091 vs 0.005–0.009)
- Insufficient regularization for 33M params on 1,388 samples
- ReduceLROnPlateau failed to prevent post-convergence overfitting
- Validation F1 plateaued at epoch 66 with subsequent fluctuations
