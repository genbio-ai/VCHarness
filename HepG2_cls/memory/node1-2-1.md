**Improvements from Parent**
- Upgraded ESM2 from 35M (480-dim direct lookup) to 650M (3840-dim embeddings)
- Added learnable 2-layer adapter (3840→1024→512→768) with 4.46M trainable parameters for ESM2-650M fusion
- Increased dropout from parent's 0.15 to 0.35 in 3-block residual MLP

**Results & Metrics (vs Parent)**
- Test F1: 0.433 vs parent 0.455 (Δ = −0.022)
- Test F1 vs tree-best (node1-1-1): −0.041
- Training: 82 epochs, severe overfitting
- Val/F1: stuck at 0.405 for epochs 1–25, peaked at 0.433 (epoch 57)
- Val/loss: frozen at ~0.268 (4–5× higher than parent's ~0.05)
- Train/loss: declined from 1.020 to 0.253

**Key Issues**
- ESM2-650M adapter (4.46M params) overfits on 1,273 training samples
- 3840-dim high-dimensional embedding space is too noisy for this dataset size
- Concatenation-based fusion dilutes STRING_GNN signal
- Adapter regularization (dropout=0.1, wd=5e-4) insufficient to prevent overfitting to irrelevant protein structural features
