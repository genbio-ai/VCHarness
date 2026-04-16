**Improvements from Parent**
- Replaced focal loss (γ=2.0) with weighted cross-entropy + label smoothing
- Reduced dropout from 0.4 to 0.35
- Changed from cosine annealing to ReduceLROnPlateau scheduler (patience=10, factor=0.5)
- Dropped ESM2 embeddings (returned to STRING_GNN-only architecture)
- Test F1 recovered from 0.157 → 0.379 (+141% relative improvement)

**Results & Metrics (vs Parent)**
- Test F1: 0.379 (vs parent 0.157, vs best sibling node1-1-1: 0.474)
- Training epochs: 138
- Validation F1 progression: 0.264 → 0.425 over 98 epochs
- LR halvings: 4 (vs 2 in node1-1-1)
- Train-val loss gap: ~0.24 (persistent throughout training)
- Val-test generalization gap: 0.047
- Output head parameters: 10.2M (75% of total 13.6M params)

**Key Issues**
- Flat 512→19,920 output head memorizes training patterns without generalizing to test data
- ReduceLROnPlateau patience=10 causes excessive LR halvings (4 halvings vs 2 in successful node1-1-1), pushing model into suboptimal local minimum
- Large train-val loss gap (~0.24) driven by oversized 10.2M-parameter output head
- Val-test gap (0.047) indicates unreliable checkpoint selection on noisy 141-sample validation set
- Identical architecture/hyperparameters to node1-1-1 produce ~0.10 F1 deficit, suggesting random variance dominates this lineage's performance ceiling
