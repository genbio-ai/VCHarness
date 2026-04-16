**Improvements from Parent**
- Replaced random perturbation embeddings with **STRING_GNN-pretrained** PPI graph embeddings (256-dim)
- Bilinear output head with **bilinear_dim=256** (vs parent 128)
- Optimizer: AdamW with **cosine annealing** (T_max=100, lr=1e-4 backbone, 2e-4 head) vs parent stepwise LR halving
- Loss: **Class-weighted CE with label smoothing** vs parent weighted CE only
- Regularization: **dropout=0.4** vs parent 0.3

**Results & Metrics (vs Parent)**
- **Test F1: 0.4527** vs 0.3700 (**+22.4% relative improvement**)
- **Val F1: 0.4527** (peaked at epoch 76, early stopping)
- **Training epochs: 92** (stopped at epoch 76)
- **Overfitting: Mild** (train-val loss gap ~0.18) vs parent severe overfitting (val loss increased 1.16→1.24)
- **Training stability:** Stable with 16-epoch plateau where val F1 oscillated 0.44-0.453
- **Generalization:** Best checkpoint generalized perfectly (val F1 = test F1)

**Key Issues**
- **Domain mismatch:** STRING_GNN encodes static PPI topology, not dynamic transcriptional perturbation response
- **Architecture limitation:** Bilinear head treats each gene independently, no gene-gene regulatory relationship modeling
