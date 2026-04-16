**Improvements from Parent**
- Upgraded LoRA from r=8 (last 4 layers) to r=4 QKV (all 8 layers) for more comprehensive adaptation
- Added frozen STRING GNN PPI embeddings (256-dim) as 4th orthogonal input channel
- Changed head from dual-pooling MLP (576→256→19920) to single-stage fusion (832→384→19920)
- Replaced cross-entropy with focal loss (γ=2.0) for better class imbalance handling
- Switched from cosine annealing to ReduceLROnPlateau (patience=8, factor=0.5)
- Implemented 3-tier differential AdamW LR: backbone 3e-4, symbol 6e-4, head 9e-4
- Reduced weight_decay from 0.05 to 0.03

**Results & Metrics (vs Parent)**
- Test F1: 0.4592 (+0.014 vs parent 0.4453, -0.003 vs tree-best 0.4622)
- Ranking: 2nd in MCTS tree
- Training duration: 44+ epochs with 3 LR reductions
- Peak val_f1: ~0.230 at epoch 24 (declined afterward)
- Severe overfitting: train_loss 0.380→0.065, val_loss 0.480→0.916 (14× gap)
- val_f1 miscalibration: ~2× underestimate relative to test F1

**Key Issues**
- val_f1 metric severely miscalibrated (2× underestimate vs test F1), making ReduceLROnPlateau unreliable for early stopping
- Early overfitting with 8.1M parameters on 1,500 training samples
- val_f1 peaked at epoch 24 then declined despite 3 LR reductions
- ReduceLROnPlateau failed to recover performance after val_f1 decline
- STRING GNN confirmed as key architectural improvement, but val_f1 miscalibration prevents proper optimization guidance
