**Improvements from Parent**
- Reduced FFN fine-tuning from 4 to 2 layers (matching node3-2 architecture)
- Reduced Muon LR from 0.03 to 0.02
- Reduced AdamW head LR from 3e-4 to 2e-4
- Reduced weight_decay from 3e-2 to 2e-2
- Replaced standard cosine decay with SGDR (CosineAnnealingWarmRestarts T_0=25, T_mult=2) + 4-epoch warmup
- Added Feature-Level Mixup (α=0.2)
- Added early stopping (patience=8)

**Results & Metrics (vs Parent)**
- Test F1: 0.4281 (+0.0106 vs parent 0.4175)
- Still below best sibling node3-2 (0.4295) by 0.0014
- Peak at epoch 29 (val F1=0.4281)
- Early stopping fired at epoch 37
- SGDR restart at epoch 28 produced +0.005 jump from ~0.423

**Key Issues**
- Feature Mixup introduced feature-space noise that competes with backbone signal on minority DEG classes
- SGDR T_0=25 misaligned with model's actual peak timing (~epoch 20)
- Recovery incomplete; still trails best sibling node3-2
