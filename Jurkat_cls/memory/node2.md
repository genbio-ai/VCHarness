**Technical Implementation**
- Model: AIDO.Cell-100M with LoRA (r=16, layers 6–17 on Q/K/V)
- Input: Synthetic expression profiles (perturbed gene=0, others=1)
- Architecture: Dual-pooling head (global mean-pool + perturbed-gene positional embedding → concat → Linear)
- Loss: Focal loss (γ=2, aggressive class weights)
- Optimizer: AdamW (backbone lr=1e-4, head lr=5e-4)
- Scheduler: ReduceLROnPlateau
- Parameters: 125M

**Results & Metrics**
- Test F1=0.404
- Best val_f1=0.404 at epoch 51
- Training epochs: 67
- Baseline comparison: +3.6% over BiGRU (0.390)
- Training loss trajectory: 1.51 → 0.08
- Validation loss trajectory: 1.73 → 5.59 (monotonic increase)
- Val F1 plateau: ~0.40 from epoch 20 onward

**Key Issues**
- Severe overfitting: train loss decreased while val loss monotonically increased
- 125M-parameter model memorized training samples without generalization
- LoRA rank=16 too expressive for 1,500-sample dataset
- No validation improvement after epoch 20 despite continued training
