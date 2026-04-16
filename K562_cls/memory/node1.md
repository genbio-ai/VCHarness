**Technical Implementation**
- Architecture: from-scratch bilinear MLP baseline (gene embedding 512-dim → MLP 256→128 → bilinear head [3,6640,128])
- Loss: weighted cross-entropy
- Optimizer: AdamW (lr=5e-3, weight_decay=1e-2)
- Learning rate schedule: progressive halving (5e-3→2.5e-3→1.25e-3)
- Training samples: 1,388

**Results & Metrics**
- Test F1: 0.3700
- Validation F1: plateaued at 0.370 after epoch 15, no improvement through epoch 46
- Training loss: 1.13 → 0.86
- Validation loss: 1.16 → 1.24

**Key Issues**
- Severe overfitting: training loss decreased while validation loss increased
- Lack of pretrained genomic knowledge: random embeddings for 1,542 perturbations cannot capture gene regulatory relationships needed to predict 6,640-gene differential expression patterns
