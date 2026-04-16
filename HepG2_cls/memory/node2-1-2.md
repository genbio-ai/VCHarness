**Improvements from Parent**
- Replaced 90:10 naive concatenation with equal-dimension projection fusion (both modalities → 256-dim before 50:50 concatenation), eliminating protein embedding dominance bottleneck
- Partially unfroze STRING_GNN: last 2 GNN layers + post_mp layers now trainable (previously frozen)
- Increased weight_decay from 1e-4 to 5e-4 and added dropout=0.3 for stronger regularization
- Used differential LR groups: 5e-5 for AIDO.Protein LoRA, 1e-4 for STRING_GNN and head layers
- Reduced label_smoothing from unspecified to 0.1

**Results & Metrics (vs Parent)**
- Test F1: 0.4680 (+0.0631 vs parent 0.4049)
- Massive recovery from sibling node2-1-1 (0.2309): +0.2371
- Nearly matches STRING-only baselines (node1-1-1: 0.474), only -0.006 gap
- Best val F1 at epoch 22 (0.4724)
- Training stability: 39 epochs with no collapse (vs parent 21 epochs)
- Val loss progression: 0.3597 at epoch 22 → 0.3726 at epoch 38 (mild overfitting)
- Val-test generalization: gap of only 0.0004 (vs parent's severe overfitting where test F1 matched epoch 0 val F1)
- Trainable parameters: 64.2M on 1,273 samples

**Key Issues**
- Protein features remain noisy due to sequence truncation (512 AA limit) and domain mismatch, causing ~0.006 performance gap vs STRING-only baselines
- Mild overfitting indicated by val loss rising 0.0129 from epoch 22 to 38 despite stable training
- Model capacity (64.2M trainable params) high relative to 1,273 training samples
