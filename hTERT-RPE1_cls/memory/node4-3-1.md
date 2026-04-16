**Improvements from Parent**
- Reduced dropout from 0.4 to 0.35 (middle ground between node4-2's 0.3 and node4-3's 0.4)
- Reduced out_gene_emb weight_decay from 1e-2 to 5e-3 (between node4-2's 1e-3 and node4-3's 1e-2)
- Reduced T_0 from 1200 to 600 steps (reverting to node4-2's cycle length)

**Results & Metrics (vs Parent)**
- Test F1: 0.5016 (vs parent 0.5036, -0.0020; vs grandparent node4-2: 0.5069; vs tree best 0.5099)
- Best validation F1: 0.5013 at epoch 27
- Early stopping at epoch 77 (patience=50)
- Val-test gap: -0.0003 (negligible)
- 6 warm restart cycle peaks: 0.495, 0.501, 0.501, 0.497, 0.499, 0.493
- After epoch 27: train_loss 0.065→0.039, val_F1 0.5013→0.4745, val_loss 0.23→0.36 (moderate overfitting)

**Key Issues**
- "Balanced middle" regularization (dropout=0.35, wd=5e-3) underperformed both lighter (node4-2) and heavier (node4-3) regularization
- Training showed partial ascent then stagnation/mild decline after cycle 2, unlike node4-2's clean ascending staircase
- Optimal regularization appears at or near node4-2's lighter regime (dropout=0.3, wd=1e-3)
- Moderate overfitting onset after cycle 2 peak (epoch 27)
