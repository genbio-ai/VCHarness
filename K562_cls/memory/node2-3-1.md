**Improvements from Parent**
- Replaced raw concatenation of multi-layer summary tokens (1920-dim) with **cross-layer attention fusion** over transformer layers [16, 17, 18]
- Implemented **learned attention-weighted pooling** that outputs 640-dim (matching proven single-layer best)
- Architecture: AIDO.Cell-100M (LoRA r=8/α=16) + frozen STRING_GNN K=16 2-head (256-dim) → 896-dim → 256 hidden MLP head

**Results & Metrics (vs Parent)**
- Test F1: **0.44606** (vs parent 0.4473, regression of **−0.0012**)
- Tree best: 0.5128 (gap of **−0.067**)
- Training progression: warmup (epochs 0–10: 0.18→0.36 val F1) → active learning (epochs 11–41: 0.36→0.44) → persistent plateau (epochs 42–70: 0.4385–0.4461, std=0.0019)
- Early stopping at epoch 70/250, LR≈8.54e-5 (never entered low-LR cosine regime)
- Zero val–test generalization gap (0.446 val ≈ 0.446 test)

**Key Issues**
- **Branch-level representational failure**: Both node2-3 (concatenation) and node2-3-1 (attention pooling) converge to identical 0.44–0.45 ceiling
- Near-final AIDO.Cell-100M layers [16, 17, 18] are **highly correlated** with no complementary signal beyond layer 18 alone
- Plateau confirmed as representational ceiling (not overfitting) due to zero val–test gap
- Multi-layer fusion strategies provide no benefit over proven single-layer summary token architecture (node2-1/node2-2 lineage achieving 0.507–0.513 F1)
