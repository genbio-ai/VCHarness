**Improvements from Parent**
- Upgraded backbone from AIDO.Cell-10M to AIDO.Cell-100M (256 hidden → 640 hidden, 8 layers → 18 layers)
- Switched from direct QKV fine-tuning to LoRA (rank=16, alpha=32, all 18 layers QKV matrices)
- Added 5-epoch linear warmup before cosine annealing schedule
- Modified prediction head architecture: Linear(256→1024→6640×3) → Linear(640→1536→19920)
- Implemented perturbed-gene knockdown encoding (perturbed gene at 0.0)

**Results & Metrics (vs Parent)**
- Test F1: 0.41475 vs 0.3853 (+0.030 improvement)
- Best val F1: 0.41472 at epoch 85
- Training duration: 126 epochs vs 95 epochs (31 more epochs)
- Early stopping triggered later (patience=40 vs patience=25)
- Overfitting pattern more severe: train loss 920→228 (4× reduction) while val loss 767→1131 (+47% increase)

**Key Issues**
- Incomplete knockdown implementation: all other genes at −1.0 (missing sentinel) instead of 1.0 baseline as required
- Single-gene-position extraction bottleneck: extracting one 640-dim vector from 19,266 hidden states to predict 6,640 output positions
- Performance ceiling around F1≈0.41 despite larger backbone and higher LoRA rank
- Warmup phase delayed competitive performance by ~70 epochs compared to sibling node3-1-1
- Severe overfitting: val-train loss gap widened substantially during training
