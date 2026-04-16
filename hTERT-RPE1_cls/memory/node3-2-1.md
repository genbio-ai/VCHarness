**Improvements from Parent**
- Realistic knockdown input representation: all 19,264 genes at expression=1.0, perturbed gene at 0.0 (replacing parent's OOD single-gene activation at 1.0 with all others at −1.0)
- 4-block pre-norm deep residual MLP prediction head (640→2560→640, 4 blocks, ~26.3M params) replacing parent's 2-layer head (640→2048→6640×3)
- LoRA rank doubled: r=32, alpha=64 on all QKV matrices across 18 transformer layers vs parent's r=16, alpha=32

**Results & Metrics (vs Parent)**
- Test F1: 0.4405 vs parent 0.4096 (+0.0309, +7.5% relative)
- Best val F1: 0.4402 at epoch 32 vs parent 0.4095 at epoch 46
- Training epochs: 73 (early stopping) vs parent 87
- Train loss trajectory: 1.24→0.15 (epochs 0–32), final 0.037
- Val F1 plateau: std=0.0027 over epochs 30–72
- Val/train loss ratio: 92× at final epoch (severe calibration overfitting)
- Gap to tree best (node1-2: 0.4912): −0.0507
- Trainable params: ~28.5M of 100M total

**Key Issues**
- Single-token information bottleneck: extracts only one 640-dim embedding from perturbed gene position in `last_hidden_state`, must encode all perturbation effects across 6,640 output positions
- Missing bilinear gene-to-gene interaction pathway present in superior STRING_GNN nodes (node1-2)
- Catastrophic probability calibration overfitting (val/train loss ratio 92×) despite stable argmax-based F1
- Performance ceiling at ~0.44 F1 attributed to insufficient information capacity of single-token representation
