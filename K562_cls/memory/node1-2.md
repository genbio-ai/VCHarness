**Improvements from Parent**
- Frozen STRING_GNN backbone with pre-computed PPI neighborhood attention aggregation (K=16, attention_dim=64, center-context gating)
- Flat 2-layer MLP head + bilinear gene-class embedding head
- Weighted cross-entropy + label smoothing (ε=0.05)
- Cosine annealing LR schedule (warmup=20 epochs, T_max=150 epochs)

**Results & Metrics (vs Parent)**
- Test F1: 0.4769 vs parent 0.3700 (+0.1069 improvement)
- Val F1: 0.4769 at epoch 70 (zero val-test generalization gap)
- Training duration: 79 epochs with smooth convergence
- Loss behavior: train loss ~0.67, val loss ~0.86 (no overfitting)
- Early stopping: never fired (val F1 remained within 1e-4 of best for 9 consecutive epochs)
- Ranking: 2nd best in MCTS tree (0.0077 behind best node F1=0.4846)

**Key Issues**
- Frozen backbone limits neighborhood attention to static PPI embeddings rather than task-adapted representations
- Confirmed performance ceiling ~0.485 for frozen STRING_GNN-only architecture
- Best MCTS result (node1-1-1-1-1, F1=0.4846) used discriminative backbone LR (1e-5) allowing task-specific node embedding refinement
