**Technical Implementation**
- STRING_GNN pretrained on STRING v12 human PPI graph (18,870 nodes) with frozen backbone
- Attention-pooled top-20 PPI neighbors feeding into 2-layer MLP head
- 3-class differential expression prediction across 6,640 genes
- Aggressive class weights: [28.1, 1.05, 90.9]
- Label smoothing: 0.10
- No focal loss implemented

**Results & Metrics**
- Test F1: 0.0494 (near-random for 3-class task)
- Dataset: 95% class-0 imbalance
- Best validation F1: 0.1937 at epoch 0 (initialization)
- Validation F1 degraded to 0.0556 by epoch 149

**Key Issues**
- Frozen backbone produces identical static embeddings for same gene ID across all samples
- Shallow MLP head receives no task-discriminative input signal
- Extreme class weights causing distributional collapse toward degenerate predictions
- Test inference used worst checkpoint instead of best validation checkpoint
