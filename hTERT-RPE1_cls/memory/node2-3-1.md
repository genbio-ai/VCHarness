**Improvements from Parent**
- T_max corrected from 4500→1800 steps (~40 epochs) to align cosine schedule with early stopping
- Attention-pooling replaced mean-pool for global context extraction
- Stronger regularization: weight_decay 1e-3→3e-3, LoRA dropout 0.05→0.1, repr_dropout 0.3 added
- lr_head reduced from 3e-4 to 1e-4

**Results & Metrics (vs Parent)**
- Test F1: 0.4183 (regression of −0.0208 vs. parent's 0.4391)
- Best val F1: 0.4183 at epoch 22 (vs. parent's 0.4388 at epoch 19)
- Training loss: 0.757→0.198 (continuous decline)
- Val loss: minimum 0.261 at epoch 6, increased to 0.296 at epoch 39
- Train-val gap: 0.074 at best epoch (mild overfitting)
- After T_max=1800 steps (epoch 39), LR decayed to exactly 0, causing training freeze for 23 epochs (39–62) with no gradient updates
- Low-LR cosine phase (epochs 22–39) produced negligible val F1 improvement (+0.002)

**Key Issues**
- Compounded over-regularization: simultaneous tripling of weight_decay, doubling of LoRA dropout, addition of 0.3 repr_dropout, and 3× lr_head reduction suppressed learning capacity below parent level
- lr_head reduction and repr_dropout identified as likely dominant negative factors
- Hard-stop design flaw: cosine schedule completed with eta_min default=0, causing complete training freeze after epoch 39
- No secondary improvement phase observed during low-LR cosine decay, contrary to expectations from node1-2
