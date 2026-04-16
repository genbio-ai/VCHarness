**Technical Implementation**
- Character-level BiGRU (2-layer) + deep MLP (hidden_dim=512)
- 3-class DEG prediction for 6,640 genes per perturbation
- Focal loss (γ=2) with class weights and label smoothing
- 1,500 training samples
- AdamW optimizer with ReduceLROnPlateau scheduler
- ModelCheckpoint monitoring `val_loss` instead of `val_f1`

**Results & Metrics**
- Test F1: 0.390
- Best val_f1: 0.509 at epoch 19
- Best val_loss at epoch 7 with val_f1=0.469
- val_f1 and val_loss anti-correlated during epochs 17-22

**Key Issues**
- Critical checkpoint misalignment: ModelCheckpoint monitored `val_loss` instead of `val_f1`
- Best model (val_f1=0.509 at epoch 19) was never saved for test prediction
- Character-level encoding of gene names carries negligible biological signal
- Hard ceiling around 0.50 F1 due to limited input representation (confirmed by node2 AIDO.Cell-100M achieving val_f1=0.783)
- Focal loss re-weighting caused val_loss spikes while improving minority-class F1
