**Improvements from Parent**
- Added GenePriorBias: per-gene per-class bias initialization with 50-epoch warmup schedule
- Added SWA (Stochastic Weight Averaging): starting epoch 200 with learning rate 1e-5
- Architecture unchanged from parent: scFoundation (6L fine-tuned) + STRING_GNN (frozen, cached) + GatedFusion

**Results & Metrics (vs Parent)**
- Test F1: 0.4801 — **identical to parent** (0.4801), **-0.0035 vs sibling node4-2-1** (0.4836)
- Validation F1 peak: 0.4867 at epoch 174
- Train/val gap: ~1.15× (moderate overfitting)
- Early stopping triggered with SWA activated only at epoch 200 (~4 checkpoints averaged)
- SWA-averaged model underperformed best pre-SWA checkpoint

**Key Issues**
- SWA triggered too late (epoch 200) — only ~4 checkpoints averaged before early stopping, causing SWA model to underperform the best pre-SWA checkpoint
- GenePriorBias warmup (50 epochs) combined with late SWA start (epoch 200) left only 150 effective bias learning epochs
- Explicit exclusion of neighborhood attention (K=16) was a negative design choice — sibling node4-2-1 with neighborhood attention achieved higher F1 (0.4836)
- No improvement over parent despite two architectural additions (GenePriorBias + SWA)
