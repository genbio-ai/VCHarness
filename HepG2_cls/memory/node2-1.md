**Improvements from Parent**
- Replaced Focal Loss (γ=2.0) + aggressive inverse-frequency weights with mild sqrt-inverse weighted CrossEntropyLoss + label smoothing
- Increased LoRA rank from r=16/α=32 to r=64 with frozen backbone (prevents overfitting)
- Added frozen STRING_GNN PPI embeddings (256-dim) as auxiliary input via attention pooling fusion (2560-dim)
- Changed learning rate from 1e-4 to 5e-5 with cosine annealing schedule
- Switched from mean pooling to attention-based pooling for protein-STRING fusion

**Results & Metrics (vs Parent)**
- Test F1: 0.4049 (vs 0.0378) — **10.7× improvement**
- Val F1: 0.4669 at epoch 17 (vs 0.0396) — **11.8× improvement**
- Training stability: stable across 21 epochs with no collapse (vs parent collapse within 1 epoch)
- Best checkpoint: epoch 17 (val F1=0.4669)
- Generalization gap: test F1=0.4049 ≈ epoch 0 val F1=0.4048, indicating val improvements do not transfer

**Key Issues**
- Protein sequence truncation to 512 AA destroys functional signal (proteins truncated before full context captured)
- Protein embeddings dominate fusion vector (90% of 2560-dim), potentially diluting informative STRING_GNN PPI signal
- Significant generalization gap: validation gains beyond initial checkpoint fail to transfer to test set
- Underperforms STRING_GNN-only baselines (node1-1: 0.472, node1-1-1: 0.474) by 0.067 F1 points
- Small training set (1,273 samples) limits model capacity
