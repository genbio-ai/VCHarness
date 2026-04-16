**Improvements from Parent**
- Switched from character-only BiGRU to **AIDO.Cell-10M backbone** with LoRA fine-tuning (rank=8, Q/K/V matrices, last 6 of 8 transformer layers)
- Added **dual pooling** architecture: gene-specific positional hidden states + global mean-pooling (512-dim)
- Replaced simple BiGRU with **multi-scale character-level CNN** on gene symbols
- Implemented **two-stage MLP head** (640→256→19920) vs single deep MLP (512)
- Fixed **ModelCheckpoint monitoring**: switched from `val_loss` to `val_f1` (mode=max)
- Enhanced regularization with **LoRA-constrained fine-tuning** vs training from scratch

**Results & Metrics (vs Parent)**
- **Test F1**: 0.4344 (+0.045 vs parent 0.390)
- **Best val_f1**: 0.5142 at epoch 43 (vs parent 0.509 at epoch 19)
- **Training dynamics**: train_loss 0.292→0.111 (-62%), val_loss 0.278→0.463 (+67%) indicating severe overfitting
- **Val-test gap**: 0.080 (0.5142 val_f1 vs 0.4344 test_f1)
- **Tree ranking**: -0.011 vs tree-best LoRA (node2-2: 0.4453), -0.028 vs tree-best overall (node3-2: 0.462)

**Key Issues**
- **Excessive LoRA capacity**: 6 layers × rank=8 on 1,500 samples ≈600 parameters/sample enables memorization
- **Severe overfitting**: train_loss decreased while val_loss increased simultaneously (67% rise)
- **Unreliable checkpoint selection**: val_f1-val_loss anti-correlation from focal loss makes val_f1 monitoring unstable
- **Val-test generalization gap**: 0.080 difference indicates checkpoint selection not aligning with true test performance
- **Inferior to tree-best**: Falls below both LoRA-best (node2-2) and overall-best (node3-2 with frozen backbone + STRING PPI fusion)
- **Focal loss side effects**: Re-weighting dominant class increases cross-entropy, causing val_f1-val_loss divergence
