**Improvements from Parent**
- Extended LoRA from QKV-only to QKV+FFN on last 4 of 8 layers (adds ~86K params)
- Switched from CosineAnnealingLR to ReduceLROnPlateau (patience=5, factor=0.7)
- Increased backbone regularization: weight_decay 0.02→0.04, head_dropout 0.4→0.5
- Reduced focal loss gamma: 2.0→1.5
- Reduced backbone_lr: 5e-4→2e-4

**Results & Metrics (vs Parent)**
- Test F1: 0.4215 vs parent 0.4453 (-0.024 regression)
- Best val_f1: 0.4215 at epoch 19 vs parent 0.445 at epoch 18 (-0.0235)
- Worst score in entire node2-2 lineage (vs best sibling 0.4472, -0.026)
- Training duration: 55 epochs (3 LR reductions triggered)
- Final overfitting gap: train_loss 0.097 vs val_loss 1.08 (11.1×)
- Val_f1 oscillated around ~0.41 throughout training without recovery

**Key Issues**
- Severe overfitting: 11.1× train-val loss gap, worst in node2-2 lineage
- Optimization failure: val_f1 never recovered epoch 19 peak despite 3 LR reductions
- Incorrect hyperparameter scaling for FFN LoRA: backbone_lr=2e-4 too low for ~86K additional params
- Over-regularized classification head: head_dropout=0.5 too aggressive
- Reduced minority-class focus: gamma=1.5 decreased from proven gamma=2.0
- Combined effect prevented learning of generalizable patterns, resulting in worst node2-2 performance
