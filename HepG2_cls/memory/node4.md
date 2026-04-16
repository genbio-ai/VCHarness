**Technical Implementation**
- ESM2-650M + STRING_GNN dual-branch architecture with gated fusion
- 3-class DEG classification task across 6,640 genes
- Full fine-tuning with aggressive learning rates

**Results & Metrics**
- Test F1: 0.4740 (highest in MCTS tree vs next best 0.2934)
- Best validation F1: 0.5005 at epoch 18
- Training loss: 0.0099
- Validation loss: 0.0771 (increasing after peak)
- Training samples: 1,273

**Key Issues**
- Overfitting: training loss continued decreasing while validation loss increased after peak
- Excessive model capacity (ESM2-650M) relative to small training set (1,273 samples)
- Learning rates too aggressive for full fine-tuning approach
