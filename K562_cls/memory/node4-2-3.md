**Improvements from Parent**
- Combined PPI Neighborhood Attention (K=16, attn_dim=64, +229K params) and GenePriorBias (warmup=30 epochs, +20K params)
- Same base architecture: scFoundation+STRING_GNN fusion with 6 fine-tuned scFoundation layers
- Same training: weighted CE + label_smoothing=0.1 + Mixup(0.2) + WarmupCosine(min_lr_ratio=0.10)

**Results & Metrics (vs Parent)**
- Test F1: 0.4688 (vs parent 0.4801) — regression -0.0113
- Best val F1: 0.4688 at epoch 134 (vs parent 0.4801 at epoch 139)
- Training: 170 epochs (early stopping)
- Sibling comparison: -0.0148 vs node4-2-1 (0.4836)
- Post-peak val F1 decline (not plateau) — overfitting signal

**Key Issues**
- GenePriorBias and neighborhood attention do NOT combine synergistically
- Each individually improves or maintains performance (node4-2-1: +0.0035; node4-2-2: neutral)
- GenePriorBias learns miscalibrated corrections on neighborhood-attended fusion features
- Only 30-epoch warmup for GenePriorBias
- Very low LR (11% of peak) at best checkpoint
