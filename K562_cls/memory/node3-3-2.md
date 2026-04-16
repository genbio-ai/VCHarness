**Improvements from Parent**
- Reduced FFN fine-tuning from 4 to 2 transformer layers (decreased capacity)
- Lowered Muon LR from 0.03 to 0.02, head LR from 3e-4 to 2e-4, weight_decay from 3e-2 to 2e-2
- Replaced cosine LR with SGDR (CosineAnnealingWarmRestarts T_0=18, T_mult=2, 5-epoch warmup)
- Added class-frequency weights to label-smoothed CE loss
- Added 3% label noise injection (DEG flip regularization)
- Adjusted early stopping: patience=10, min_delta=0.0005

**Results & Metrics (vs Parent)**
- Test F1: 0.4496 vs 0.4175 (+0.0321, +7.7%)
- Val peak: 0.4496 at epoch 42 vs 0.4175 at epoch 25
- Training duration: 53 epochs vs 45 epochs (ES fired later)
- SGDR cycle 1 length: 36 epochs (vs parent's continuous cosine decay)
- F1 climb from restart trough: 0.037 (0.413→0.450)

**Key Issues**
- Parent model stagnated after epoch 25 with cosine LR; SGDR restarts enabled sustained improvement
- Parent's higher LR (0.03) + expanded FFN (4 layers) caused overfitting; reduced capacity and LR stabilized training
- Short cycle 1 (12 epochs) in sibling node3-3-1 limited exploration; T_0=18 produced optimal 36-epoch first cycle
