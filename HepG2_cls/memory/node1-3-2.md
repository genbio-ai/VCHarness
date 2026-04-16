**Improvements from Parent**
- Removed ESM2-35M protein sequence embeddings (parent used additive fusion STRING_GNN + ESM2)
- Reduced hidden_dim from 512 to 384 (parent: 512)
- Changed to flat output head (Linear 384→19920) with per-gene bias
- Added Muon optimizer for hidden block weight matrices (lr=0.02) + AdamW (lr=3e-4) for other params (parent: AdamW only, lr=3e-4)
- Added gradient clipping (max_norm=1.0)
- Increased ReduceLROnPlateau patience from 5 to 8

**Results & Metrics (vs Parent)**
- Test F1: 0.4756 (parent: 0.463) — +0.013 improvement
- New tree best, surpassing node1-1-1's previous ceiling of 0.474 (+0.002)
- Best checkpoint at epoch 75: val/f1=0.4753, train/loss=0.211, val/loss=0.275
- Training completed 100 epochs (parent: 83 epochs, early stopped)
- Train-val loss gap: 0.064 (parent: 0.026)
- Loss ratio: 0.064/0.275 = 23% (parent: 0.026/0.258 = 10%)

**Key Issues**
- Persistent train-val loss gap (0.064) indicating continued overfitting, though reduced in relative terms compared to 512-dim nodes
- Muon optimizer showed stable convergence without fluctuations (contrast to sibling node1-3-1)
