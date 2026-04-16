**Improvements from Parent**
- Corrected LR schedule T_max to 1800 steps (~40 epochs) vs parent's ~4500 steps, ensuring full cosine completion
- Set eta_min=1e-7 to prevent hard LR=0 freeze during post-T_max plateau
- Reduced head LR from 3e-4 to 2e-4

**Results & Metrics (vs Parent)**
- Test F1: 0.4296 vs parent 0.4391 (Δ=-0.0095)
- Best val F1: 0.4295 at epoch 22 (mid-cosine phase, backbone LR at 43% of peak)
- Outperformed siblings node2-3-1 (0.4183) and node2-3-1-1 (0.4309)
- Below tree best node1-2-2-2 (0.5060)
- Post-T_max plateau (epochs 40–62): val F1 std=0.000083 (near-zero variance)
- Train-val loss gap: +0.112 at best epoch 22, widening to +0.139 at epoch 62

**Key Issues**
- Head LR reduction from 3e-4 to 2e-4 measurably hurt performance vs parent
- Secondary improvement hypothesis (cosine completion unlocks second learning phase) not supported — architecture reaches optimum at mid-cosine
- eta_min=1e-7 produced no effective updates during post-T_max plateau
- AIDO.Cell branch performance ceiling appears limited near F1~0.44
