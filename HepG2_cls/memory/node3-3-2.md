**Improvements from Parent**
- Reduced from 4-block to 3-block PreNorm residual MLP architecture
- Changed hidden dimension from 512 to 384
- Changed RLROP patience from 10 to 8
- Removed 5-epoch warmup
- Added head_dropout=0.05 (vs no head dropout in parent)

**Results & Metrics (vs Parent)**
- Test F1: 0.4536 (+0.067 vs parent node3-3 at 0.387, +0.031 vs sibling node3-3-1 at 0.4226)
- Best val F1: 0.4541 at epoch 62
- Train loss: ~1.0 at best checkpoint (4.5× worse than node1-3-2 at 0.211, 80× worse than node1-1-1 at 0.012)
- Training: 182 epochs with RLROP halvings at epochs 70 and 163
- Val F1 declined from 0.4541 to 0.381 by epoch 182 despite train loss reduction (1.0→0.93)
- RLROP halvings produced only marginal +0.002–0.004 F1 recovery

**Key Issues**
- Pure AdamW optimizer with h=384 unable to fit training distribution (train_loss≈1.0)
- Model trapped in underfitting basin — plateau-then-decline pattern with continued train loss reduction but worsening val F1
- STRING-only ceiling for AdamW configuration conclusively F1≈0.454
- Train/loss 4.5×-80× worse than tree-best nodes using Muon or AdamW at h=512
