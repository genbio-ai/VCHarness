**Improvements from Parent**
- Expanded FFN fine-tuning from 2 to 4 layers in last 4 transformer layers (1.57M new params)
- Increased Muon LR from 0.02 to 0.03
- Increased weight_decay from 2e-2 to 3e-2
- Added SWA planning (epoch 220)

**Results & Metrics (vs Parent)**
- Test F1: 0.4175 vs parent 0.426 = -2.0% regression
- vs best sibling node3-2 (0.4295): -2.8% regression
- Peak at epoch 25 (val F1=0.4175)
- Early stopping fired at epoch 45 (before SWA phase at 220)

**Key Issues**
- Combination of expanded FFN capacity (4 vs 2 layers) and higher Muon LR (0.03 vs 0.02) pushed optimizer into worse local minimum
- Higher LR amplified FFN gradient noise through Muon's Newton-Schulz orthogonalization
- Doubled FFN capacity added overfitting-prone degrees of freedom
- Model never recovered despite 20 additional epochs of training and cosine LR decay
