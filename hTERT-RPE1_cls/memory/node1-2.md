**Improvements from Parent**
- Replaced random nn.Embedding with frozen STRING_GNN providing 256-dim pretrained PPI embeddings from 393,006 protein-protein interaction edges
- Upgraded from 4-layer to 6-layer residual MLP with deeper architecture (hidden_dim=512, expand=4, Dropout=0.2)
- Added bilinear interaction layer between PPI embeddings and learnable output-gene embeddings [6640, 256] replacing low-rank output head
- Switched from weighted cross-entropy to focal loss (γ=2.0) to handle 88.9% neutral class imbalance
- Replaced ReduceLROnPlateau with cosine annealing scheduler using 50-step warmup (lr=5e-4, wd=1e-3)

**Results & Metrics (vs Parent)**
- Test F1: **0.4912** vs parent 0.3762 (+0.1150, +30.6%)
- Best val F1: 0.4911 at epoch 98
- Training duration: 129 epochs vs parent 81 epochs
- Val loss trajectory: min 0.1114 at epoch 11, increased to 0.1470 at epoch 129 (vs parent: min at epoch 16)
- Final val/train loss ratio: 3.71× vs parent ~11× (less severe overfitting)
- Outperformed all nodes: +0.1823 vs node1-1 (0.3089), +0.0654 vs node4 (0.4258)

**Key Issues**
- Frozen STRING_GNN backbone provides static PPI embeddings that cannot capture perturbation-specific signal propagation through the network
- Val loss increased monotonically from epoch 11 onward (0.1114→0.1470) indicating calibration overfitting despite F1 improvement
- Training showed plateau phases: epochs 20–70 (F1 stalled at ~0.45), secondary improvement only during LR decay (epochs 70–98)
