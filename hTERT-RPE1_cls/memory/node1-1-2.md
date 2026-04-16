**Improvements from Parent**
- Switched from AIDO.Cell-10M backbone to frozen STRING_GNN (256-dim) + frozen ESM2-35M (480-dim) dual-source fusion
- Replaced gene-specific embedding extraction with fixed protein topology + sequence features
- Upgraded to 6-layer deep residual bilinear MLP head (hidden_dim=512, rank=256)
- Added label_smoothing=0.05 to focal loss (gamma=2.0 maintained)
- Changed optimizer to AdamW (lr=5e-4, weight_decay=1e-3) from separate backbone/head LR schedule
- Increased patience to 60 (from 45 epochs total in parent)

**Results & Metrics (vs Parent)**
- Test F1: **0.4822** vs parent 0.3089 (+0.1733 improvement)
- Best val F1: 0.4820 at epoch 121 (vs parent 0.3089 at epoch 30)
- Training completed 181 epochs with early stopping at patience=60
- Val F1 stability: oscillated in [0.4741, 0.4820] during plateau (std=0.00135)
- No train-val loss divergence indicative of classical overfitting (unlike parent)
- Sibling node1-1-1: F1=0.3669, tree best node1-2: F1=0.4912

**Key Issues**
- ESM2 protein sequence embeddings (480-dim) added noise rather than signal for transcriptional perturbation prediction
- Performance fell 0.009 below STRING_GNN-only node1-2 baseline despite larger input dimension (256→736)
- Larger proj_in layer (736→512 vs 256→512) requires learning significantly more parameters from only 1,416 samples
- Protein sequence features less informative than PPI topology for predicting downstream DEG signatures
- Val loss diverged from val F1 after epoch 26 (characteristic of focal loss dynamics)
- Model reached representational capacity ceiling with stable plateau (epochs 121-181)
