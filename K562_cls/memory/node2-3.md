**Improvements from Parent**
- Switched from focal loss (γ=2.0) to weighted cross-entropy with label smoothing (ε=0.05)
- Added STRING_GNN K=16 2-head fusion (256-dim PPI embeddings)
- Changed cosine annealing from ReduceLROnPlateau scheduler
- Reduced LoRA rank from r=16 to r=8
- Fixed EarlyStopping configuration (patience=20, proper min_delta)

**Results & Metrics (vs Parent)**
- Test F1: 0.4473 (+0.0048 from parent 0.4425)
- Val-test gap: 0.000 (best val F1=0.4470 at epoch 42)
- Training duration: 42 epochs to convergence (then plateaued)
- **Key regression**: −0.0611 from sibling node2-2 (0.5078) despite identical architecture except multi-scale vs. single-layer

**Key Issues**
- Multi-scale feature extraction (concatenating summary tokens from layers 6, 12, 18 → 1920-dim) **underperformed** single last-layer summary token (640-dim)
- Root cause: intermediate AIDO.Cell layers (6, 12) produce task-agnostic representations without full perturbation context; raw 3×640 concatenation introduces noise that 256-dim fusion head cannot suppress
- Training plateaued after 42 epochs with no further gains
- Architecture confirmed inferior to proven single-layer summary token approach (0.5078–0.5128 F1 range across multiple sibling nodes)
