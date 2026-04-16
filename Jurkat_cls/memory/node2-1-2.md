**Improvements from Parent**
- Three-source feature fusion architecture: AIDO.Cell-10M backbone (LoRA on QKV of last 4 layers) + character-level 3-branch gene symbol CNN + frozen STRING GNN PPI embeddings
- Fusion head: 704→320→19920 dimensions
- Differential learning rates: backbone 3e-4, fusion components 1.5e-3
- ReduceLROnPlateau scheduler (patience=5, factor=0.7) replacing OneCycleLR
- Focal loss (γ=2, weights=[3.0,1.0,7.0]) replacing standard cross-entropy
- Training on 1,500 samples (vs parent's 6,640 genes dataset)

**Results & Metrics (vs Parent)**
- Test F1: 0.4453 vs parent 0.4101 (+0.0352)
- Matches node2-2's score exactly (0.4453)
- Best checkpoint at epoch 32: val_f1=0.445, train_loss=0.057, val_loss=0.435
- Validation F1 trajectory: 0.381→0.445 (monotonic improvement to epoch 32)
- Final epoch 67: train_loss=0.035, val_loss=0.573, val_f1=0.441
- 5 LR reductions failed to improve upon epoch-32 peak
- Overfitting gap: 7.6× at best (val_loss/train_loss), 16× at end

**Key Issues**
- Synthetic one-hot-like expression input (perturbed_gene=0, others=1.0) constrains all three feature sources to same coarse positional signal
- Additional feature capacity (symbol CNN + STRING GNN) provides orthogonal gene-identity signal but cannot fundamentally break through current performance ceiling
- Falls short of tree-best (node3-2=0.4622) and node2-2-1=0.4472
- Val_loss increased from 0.319→0.435 while val_f1 improved, indicating calibration issues
