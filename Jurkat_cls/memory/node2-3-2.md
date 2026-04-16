**Improvements from Parent**
- LoRA rank reduced from r=8 (last 4 layers) to r=4 (all 8 layers) for broader adaptation coverage
- Added frozen STRING GNN PPI embeddings (256-dim) as fourth input source
- Gene symbol CNN expanded to 64-dim (up from architecture in parent)
- Fusion architecture redesigned: 4-source concat (832-dim) → 384 → 19920 output head
- ReduceLROnPlateau patience tightened from 8→5, factor increased 0.7→0.5 for faster LR response
- Focal loss gamma increased 1.5→2.0 for harder example focus
- Class weights set to [5, 1, 10] for minority class emphasis
- Optimizer learning rates differentiated: backbone_lr=2e-4, head_lr=6e-4 (3:1 ratio)

**Results & Metrics (vs Parent)**
- Test F1: 0.4450 (+0.0075 vs parent node2-3: 0.4375)
- Test F1: +0.0092 vs sibling node2-3-1 (0.4358)
- Best val_f1: 0.4450 at epoch 27 (LR reduction point)
- Near-zero val-test gap indicates minimal overfit
- Training epochs: 47 total with controlled degradation under LR starvation after epoch 27
- Reference node2-3-1-1 achieved 0.4555 with class weights [7, 1, 15] (+0.0105 gap)

**Key Issues**
- Class weights [5, 1, 10] too conservative vs proven [7, 1, 15] configuration from node2-3-1-1
- Weaker minority class emphasis reduces per-gene macro F1 for rare differential expression events
- STRING GNN confirmed as primary improvement driver but underutilized due to suboptimal class weighting
- Checkpoint averaging not yet implemented (reference node2-3-1-1 used top-3 averaging)
