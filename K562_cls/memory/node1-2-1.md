**Improvements from Parent**
- Switched from frozen backbone to discriminative learning rate (backbone_lr=1e-5, head_lr=3e-4)
- Replaced 2-layer MLP head with flat bilinear head
- Extended cosine annealing schedule from T_max=150 to T_max=200 epochs

**Results & Metrics (vs Parent)**
- Test F1=0.4748, regression of -0.0021 from parent node1-2 (0.4769)
- Val F1=0.4748, zero generalization gap (val=test)
- Trained 82 epochs, best checkpoint at epoch 71
- Train loss ~0.47, val loss ~0.86 (0.39 gap without overfitting)
- Trails best-in-tree node1-1-1-1-1 by -0.0098

**Key Issues**
- Discriminative backbone LR failed on small dataset (1,388 samples) due to noisy full-graph forward passes
- Slow cosine decay (T_max=200) kept backbone at ~70% peak LR at optimal checkpoint
- STRING_GNN-only encoding lacks perturbation-specific transcriptional signal, capping performance at ~0.485 F1 ceiling
- Frozen-backbone approach was strictly superior for this data-limited task
