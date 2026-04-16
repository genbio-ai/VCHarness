**Improvements from Parent**
- Replaced mean-pool with summary-token feature extraction
- Reduced classification head to 256-dim with dropout=0.5 (vs 512-dim, no dropout)
- LoRA rank r=8 (vs r=16)
- Weighted cross-entropy + label smoothing ε=0.05 (replacing focal loss γ=2.0)
- Cosine annealing LR scheduler with 10-epoch warmup (replacing ReduceLROnPlateau)
- Added weight decay=2e-2
- Early stopping enabled (patience not specified in memory)

**Results & Metrics (vs Parent)**
- Test F1=0.4670 vs parent 0.4425 (+0.025 improvement)
- Best result in AIDO.Cell lineage, ranked 2nd across full tree (after node1-1-1-1: 0.4746)
- Trained 80 epochs, peak at epoch 74
- Val F1 progression: 0.20 → 0.467
- Zero val-test generalization gap at peak
- Persistent train-val loss gap of ~0.21 indicating moderate overfitting

**Key Issues**
- Moderate overfitting persists (train-val loss gap ~0.21)
- Domain mismatch: AIDO.Cell trained on steady-state transcriptomics, task requires perturbation propagation awareness
- AIDO.Cell lacks mechanistic understanding of regulatory network perturbations
- STRING_GNN outperforms by ~0.008 F1 due to PPI topology being more domain-relevant
