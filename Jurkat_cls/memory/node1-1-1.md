**Improvements from Parent**
- Added LoRA fine-tuning to AIDO.Cell-100M backbone (r=16, Q/K/V on layers 6-17, ~0.74M params)
- Replaced global mean-pool with dual pooling (concatenates perturbed gene's positional hidden state with global mean-pool → 1280-dim)
- Kept 512-dim MLP head classifier
- Total trainable params: 11.6M
- Used focal loss with class weights [5.0, 1.0, 10.0] and cosine LR annealing

**Results & Metrics (vs Parent)**
- Best val_f1=0.411 at epoch 27 (parent node1-1: best val_f1=0.469 at epoch 211, test_f1=0.390)
- Test F1=0.411 (+0.021 vs parent's 0.390)
- Training pattern: train_loss 0.416→0.186, val_loss 0.388→0.472 (severe overfitting after epoch 27)
- Performance gap: +0.021 over frozen backbone parent, -0.372 below node2's val_f1=0.783
- Model capacity: 11.6M trainable params for 1,500 samples (~7,733 params/sample)

**Key Issues**
- Severe overfitting: val_loss increased while train_loss decreased, indicating memorization
- Excessive model capacity relative to data size (11.6M params for 1,500 training samples)
- Minimal improvement over frozen backbone (+0.021 test F1) despite fine-tuning
- Catastrophically below node2's performance (-0.372 F1 gap)
- Single-gene perturbation profiles with near-one-hot inputs prevent learning discriminative features without overfitting
