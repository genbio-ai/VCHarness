**Technical Implementation**
- Deep residual MLP with 19.4M parameters
- Learned gene embeddings (dimension=512, 8 residual blocks)
- Input: single perturbed gene identifier
- Output: 6,640 per-gene 3-class differential expression predictions
- Loss: weighted cross-entropy with label smoothing
- Training samples: 1,273

**Results & Metrics**
- Test F1: 0.405
- Best checkpoint: epoch 8 (val/f1=0.405)
- Training loss: 0.303→0.248 (continued decreasing)
- Val loss and F1: flat at 0.405 after epoch 8

**Key Issues**
- Mild overfitting: training loss decreases while validation metrics plateau
- Hard performance ceiling at val F1=0.405
- Architectural insufficiency: model must learn entire gene interaction network from scratch using only one gene ID as input
- No biological prior knowledge incorporated
