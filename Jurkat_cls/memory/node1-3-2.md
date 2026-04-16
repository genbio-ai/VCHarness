**Improvements from Parent**
- Reduced LoRA rank from 8 to 4 across all 8 transformer layers (vs parent's rank-8 on last 6 layers only)
- Simplified head architecture from two-stage (640→256→19920) to single-stage (832→384→19920)
- Increased class weight aggressiveness: [5.0, 1.0, 10.0] (vs parent's unspecified weights)
- Elevated head learning rate to 9e-4 (3× backbone 3e-4) for differential optimization
- Added top-3 checkpoint averaging across epochs 28/36/42

**Results & Metrics (vs Parent)**
- Test F1: 0.4344 (identical to parent node1-3, -0.008 vs sibling node1-3-1)
- Val F1: 0.5374 at epoch 28 (+0.023 vs parent's 0.5142)
- Training loss: 0.392→0.068 (-82% from epoch 0 to 48)
- Val loss: 0.500→0.980 (+96% from epoch 0 to 48)
- Val-test inversion: +0.103 overestimation (opposite of parent's expected underestimation pattern)

**Key Issues**
- **Critical val-test anti-correlation**: val_f1=0.5374 overestimated test F1=0.4344 by +0.103 — exact opposite of documented LoRA+STRING regime expectation
- **Severe overfitting**: 82% train_loss reduction vs 96% val_loss increase; val_loss trajectory 0.50→0.98
- **Aggressive class weights failure**: [5.0, 1.0, 10.0] with focal loss causes heavy minority-class prediction on train/val distribution without test transfer
- **Head LR too high**: 9e-4 (3× backbone) accelerates overfitting on only 1,500 samples
- **Checkpoint averaging reinforced memorization**: epochs 28/36/42 all from overfitting regime (val_loss 0.81–0.88)
- **STRING adds zero value**: 0% improvement over no-STRING parent node1-3 (both at 0.4344)
