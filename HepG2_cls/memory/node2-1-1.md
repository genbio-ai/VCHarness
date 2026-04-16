**Improvements from Parent**
- Increased sequence length from 512 to 1024 tokens
- Added learned element-wise gated fusion mechanism (256→2304 projection + 2560→2304 gate)
- Increased weight decay from 1e-4 to 0.01 (100× stronger)

**Results & Metrics (vs Parent)**
- Test F1: 0.2309 vs parent 0.4049 (Δ = -0.174, catastrophic regression)
- Best val F1: 0.4524 at epoch 5 (val F1 at epoch 0: 0.4049)
- Test predictions collapsed to uniform random: 33%/33%/33% per class
- Worst test F1 across entire search tree

**Key Issues**
- Over-regularization: wd=0.01 is 100× stronger than proven baselines (wd=1e-3)
- Gating mechanism added 5.9M parameters, causing capacity collapse in large 16B encoder
- Checkpoint loading bug: test evaluation used epoch 0 checkpoint (val F1=0.4049) instead of best epoch 5 checkpoint (val F1=0.4524)
- Training instability: val F1 peaked at epoch 5 but model collapsed by final evaluation
