**Improvements from Parent**
- Replaced SWA averaging mechanism with standard ModelCheckpoint (save_top_k=3)
- Maintained scFoundation top-6L fine-tuning with gradient checkpointing + frozen STRING_GNN + GatedFusion + GenePriorBias architecture

**Results & Metrics (vs Parent)**
- Training: 254 epochs, best val/f1=0.4884 at epoch 229 (vs parent ~0.487 at epoch 171)
- Training stability: healthy plateau with std=0.0023, no overfitting
- Test F1: 0.3532 (catastrophic collapse vs parent 0.4585, vs best sibling 0.4801)
- Test degradation magnitude: -0.1052 vs parent, -0.1269 vs best sibling

**Key Issues**
- Broken ModelCheckpoint filename pattern: `{val_f1:.4f}` uses underscore but logged metric is `val/f1` (forward slash)
- All checkpoint filenames showed val_f1=0.0000, causing save_top_k=3 to save last 3 epochs (229, 250, 253) instead of best 3
- Three near-identical late-convergence models averaged into random-level state
- Epoch-0 training loss was 1.7578, matching the collapsed test F1=0.3532 (random initialization level)
