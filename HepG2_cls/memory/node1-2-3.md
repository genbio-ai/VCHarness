**Improvements from Parent**
- Focal gamma reduced from 2.5 → 2.0
- Weight decay increased from 0.02 → 0.03 for MLP trunk
- Dedicated AdamW group for gene_bias (19,920 params) with wd=0.10 (differential weight decay)
- Head dropout increased from 0.08 → 0.15
- Manifold Mixup added (alpha=0.2, prob=0.25)
- CosineWR T_0 reduced from 80 → 40 (more frequent warm restarts)
- MuonWithAuxAdam optimizer (auxiliary AdamW for gene_bias group)

**Results & Metrics (vs Parent)**
- Test F1=0.4572 (vs parent 0.4884) — -0.0312 regression
- Best val_f1=0.5107 at epoch 94 (vs parent 0.5431 at epoch 381) — -0.0324 regression
- Train/val loss ratio=2.55× (vs parent 13.04×) — 5.1× improvement, resolved extreme memorization
- Val-test gap=0.0536 (vs parent 0.0547) — structurally identical, ~0.054 persists
- Early stopping at epoch 114 (patience=20), 6 epochs before second warm restart (epoch 120)
- Above sibling node1-2-2 (0.4433) and node1-2-1 (0.4330)

**Key Issues**
- Persistent structural val-test distribution gap (~0.054) unchanged from parent — regularization does not address this
- Collective over-regularization from 5 simultaneous mechanisms (gamma=2.0 + wd=0.03 + gene_bias wd=0.10 + dropout=0.15 + mixup prob=0.25) lowered val F1 ceiling from 0.5431 → 0.5107
- Training stopped 6 epochs before second CosineWR warm restart (epoch 120), missing potential val F1 recovery
- Lower val F1 ceiling + fixed structural gap directly caused test F1 regression
