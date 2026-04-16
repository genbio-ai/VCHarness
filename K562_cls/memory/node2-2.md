**Improvements from Parent**
- AIDO.Cell-100M LoRA reduced rank from r=16 to r=8
- Added STRING_GNN K=16 with 2-head neighborhood attention fusion
- Replaced focal loss (γ=2.0) with weighted CE + label smoothing
- Replaced ReduceLROnPlateau with cosine annealing warmup scheduler
- Increased regularization: dropout=0.5 (from unspecified), weight_decay=2e-2
- Fixed summary token feature extraction

**Results & Metrics (vs Parent)**
- Test F1: **0.5078** vs parent 0.4425 (+0.065, +14.7%)
- Test F1: +0.041 vs sibling node2-1 (0.4670)
- Training: 88 epochs, best checkpoint at epoch 83
- Train-val loss gap: ~0.14 (no severe overfitting vs parent train loss 0.0036)
- STRING_GNN PPI fusion contributed +0.04–0.05 over AIDO-only
- Ranked second-best in search tree (gap 0.005 to best node)

**Key Issues**
- Possible over-regularization from dropout=0.5 limiting fine-grained DEG capture
