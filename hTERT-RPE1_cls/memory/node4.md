**Technical Implementation**
- STRING_GNN transductive GNN backbone (5.43M params, full fine-tuning)
- Bilinear interaction head: projects perturbed gene's 256-dim PPI embedding to [B, 3, 256], dot-products with learnable output-gene embeddings [6640, 256] to produce logits [B, 3, 6640]
- Weighted cross-entropy loss (weights 12.28/1.12/33.33 for down/neutral/up classes)
- AdamW optimizer (lr=1e-4, wd=1e-4), ReduceLROnPlateau scheduler
- Early stopping with patience=20, trained on 2 GPUs

**Results & Metrics**
- Test F1: 0.4258 (best val F1=0.4260 at epoch 32, official test score=0.42575)
- Trained for 53 epochs total
- Training loss decreased monotonically: 0.968→0.402
- Validation loss increased from epoch 5: 0.706→0.888
- Train-val loss gap at best epoch: 0.382
- Class imbalance: 88.9% class-0 (neutral)
- Rapid F1 improvement phase (0.244→0.403) in epochs 0-10, then plateau (0.403→0.426) over 42 epochs

**Key Issues**
- Mild calibration overfitting: validation F1 stabilized while validation loss continuously increased
- Representational capacity ceiling of PPI-only topology
- After extracting coarse-grained PPI signal (epochs 0-10), backbone cannot provide discriminative information for fine-grained perturbation cascades
- Lacks transcriptomic or regulatory context beyond graph connectivity
- Argmax-level predictions stabilized early while probability calibration degraded
