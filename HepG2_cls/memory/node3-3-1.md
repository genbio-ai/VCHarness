**Improvements from Parent**
- Reduced from 4 blocks to 3 blocks (matching tree-optimal architecture)
- Reduced hidden dimension from h=512 to h=384 (matching tree-optimal capacity)
- Changed optimizer from pure AdamW to Muon+AdamW dual optimizer
- Reduced head dropout from 0.40 to 0.15
- Added label smoothing=0.05
- Tightened RLROP patience from 10 to 8

**Results & Metrics (vs Parent)**
- Test F1: 0.4226 (parent: 0.387) — +0.0356 improvement
- Train/loss: 0.94 (parent: 0.947) — marginally improved
- Val F1: 0.423 (plateaued)
- Ran 137 epochs with 3 RLROP halvings (parent: 182 epochs, halvings at 70/164/175)
- Test-val gap <0.001 (no generalization issue)
- Train/loss still 4.5× worse than tree-best (0.207)

**Key Issues**
- Severe training underfitting: train/loss=0.94 vs tree-best 0.207 (4.5× gap)
- Val F1 plateaued at 0.423 vs tree-best 0.4777 (0.055 gap)
- Cumulative regularization burden from Muon LR=0.02 + label smoothing=0.05 + head dropout=0.15 prevents output head from fitting training distribution
- 3-block h=384 architecture is sound, but optimization hyperparameters are over-regularized
