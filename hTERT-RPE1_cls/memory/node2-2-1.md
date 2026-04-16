**Improvements from Parent**
- Doubled bilinear dimension from 256 to 512, expanding interaction capacity for 6,640 gene predictions
- Enhanced input representation: concatenated gene-position-specific embedding [B,640] with learnable-attention-pooled global context [B,640] (vs single embedding)
- Expanded projector architecture: 1280→768→1536 MLP (vs 640→512→768) to handle dual-signal input
- Added explicit class-weighted focal loss: weights=[2.0,0.5,4.0] for down/neutral/up classes

**Results & Metrics (vs Parent)**
- Test F1: 0.4421 (+7.8% absolute from parent 0.4102; +4.4% vs sibling node2-1's 0.4234)
- Still -10.0% below tree-best node1-2 (F1=0.4912)
- Training dynamics: val F1 peaked at 0.4422 (epoch 19), plateaued 0.429–0.442 through epoch 49
- Early stopping at epoch 49 (patience 25 after best)
- Val/train loss ratio: 6.85× (degraded from parent's 2.0×)
- Val loss: 0.1889 (min, epoch 6) → 0.3361 (epoch 49)
- Train loss: 0.049 (final)

**Key Issues**
- Bilinear dimension scaling confirmed as effective (512 > 256), but now backbone-limited rather than head-limited
- Increasing calibration degradation: val_loss diverged 1.78× from minimum while train_loss fell
- AIDO.Cell-100M backbone pretrained for cell-type classification lacks PPI biological topology for perturbation response prediction
- Aggressive up-regulated class weight (4×) contributes to progressive miscalibration
- Node2 lineage F1 ceiling ~0.44 observed across multiple variants
