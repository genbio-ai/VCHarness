**Improvements from Parent**
- scFoundation fine-tuning expanded from top-4 to top-6 layers
- Loss changed from focal loss with class weights to weighted cross-entropy with label smoothing
- Added Mixup augmentation for training regularization
- Added GenePriorBias to classification head
- STRING_GNN changed from full fine-tuning to frozen
- Achieved stable training with 0.95x train/val loss ratio (vs 18x gap in parent)

**Results & Metrics (vs Parent)**
- Best validation F1: ~0.487 at epoch 171 (vs ~0.46 in parent)
- Test F1: 0.4585 (identical to parent, 0.0217 below best sibling node4-2)
- SWA averaged 68 checkpoints from epoch 160-227
- Test F1 degraded by ~0.0285 from best validation checkpoint due to SWA

**Key Issues**
- SWA catastrophically degraded test performance: 68 checkpoints averaged across epochs 160-227 destroyed task-specific patterns
- Root cause: over-averaging too many diverse model states in complex multi-layer transformer fused with gene-level classification head
- Node4-2-1-1-1 success with only 8 SWA checkpoints confirms fewer snapshots required
- Recommended fix: remove SWA entirely and use best checkpoint (~epoch 171), or limit SWA to final 8-10 epochs (start at epoch 220)
