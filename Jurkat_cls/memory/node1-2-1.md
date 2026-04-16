**Improvements from Parent**
- Replaced dynamic STRING_GNN conditioning with pre-computed frozen static embeddings (256-dim)
- Added third input channel: character-level 3-branch CNN on Ensembl gene IDs (64-dim)
- Expanded fusion dimension from 768→832 to accommodate additional CNN branch
- Increased head intermediate layer from 128→256 dimensions
- Architecture: AIDO.Cell-10M dual-pool (512-dim) + frozen STRING static (256-dim) + char-CNN (64-dim) → MLP 832→256→19920

**Results & Metrics (vs Parent)**
- Test F1=0.4214 vs parent 0.3896 (+0.032 improvement)
- Zero val-test gap (excellent generalization, test equals val)
- Monotonic val_f1 improvement to plateau at epoch 24
- Mild val_loss creep after epoch 24 while val_f1 stable
- Performance ceiling: ~0.42 F1, below node2-2 lineage (0.4472)
- Healthiest training profile in tree (no overfitting)

**Key Issues**
- Frozen backbone provides only positional lookup from synthetic input
- Synthetic one-hot input encoding creates hard representational ceiling
- Char-CNN and STRING embeddings add orthogonal but insufficient signal to break ceiling
- Combined frozen features cannot escape 0.42 F1 barrier despite optimal generalization
