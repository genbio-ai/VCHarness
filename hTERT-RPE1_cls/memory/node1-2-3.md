**Improvements from Parent**
- Increased bilinear output interaction head rank from 256 to 512 (+2.1M parameters)
- Added class-weighted focal loss with weights [down=2.0, neutral=0.5, up=4.0] (gamma=2.0)
- Applied selective weight decay exempting out_gene_emb embedding table
- Used calibrated cosine LR schedule with total_steps=4000 (vs parent's 6600) and patience=50

**Results & Metrics (vs Parent)**
- Test F1: 0.4969 vs parent 0.4912 (+0.0057, +1.16%)
- Best val F1: 0.4969 at epoch 25 (step 571, LR=4.79×10⁻⁴)
- Outperformed siblings: node1-2-1 (0.4500), node1-2-2 (0.4664)
- Training duration: 75 epochs (early stopping)
- Training phases: rapid improvement epochs 0–16 (F1: 0.3352→0.4942), peak at epoch 25, broad plateau until epoch 75 (val_f1 std=0.0047)
- Train loss: 0.3607→0.0497 (decreasing)
- Val loss: 0.2495→0.2670 (+57.2% from minimum at epoch 6=0.1698)

**Key Issues**
- Best checkpoint occurred at 14.3% through cosine schedule (step 571/4000) — premature convergence before low-LR fine-tuning phase
- Expected secondary improvement phase (analogous to parent's epoch 70–98 gain) never materialized
- Progressive calibration overfitting: val loss increased +57.2% from epoch 6 minimum while train loss continued decreasing
- Class-weighted focal loss disrupted neutral-class gradient balance, creating noisier early optimization landscape that forced model into suboptimal basin
