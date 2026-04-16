**Improvements from Parent**
- Swapped from AIDO.Protein-16B protein sequence encoder to AIDO.Cell-10M single-cell encoder
- Replaced protein language model inputs with frozen STRING GNN PPI embeddings (256-dim) + 3-branch symbol CNN (64-dim)
- Changed LoRA configuration from r=8 last-4 layers to r=8 last-4 of 8 layers (alpha=16)
- Switched from generic DEG prediction focal loss to focal loss with ReduceLROnPlateau on val_loss (patience=5, factor=0.5)
- Added single-stage MLP head: 832→384→19920

**Results & Metrics (vs Parent)**
- Test F1: 0.4513 vs parent 0.0498 (+0.4015, ~9× improvement)
- Best val_f1: 0.4513 at epoch 13
- Training duration: 38 epochs vs parent 28 epochs
- Val-test gap: 0.0 (perfect generalization) vs parent not reported
- MCTS rank: 2nd overall, −0.011 below tree-best node3-2 (0.462)
- LR reductions: 4 successful reductions (93.75% total decay)
- Overfitting progression: train-val gap escalated from 3.87× to 6.43× post-peak

**Key Issues**
- LoRA r=8 last-4 configuration underperforms r=4 all-8-layers by −0.011 F1
- Synthetic one-hot-like input paradigm has hard ceiling at ~0.462 F1, confirmed by 10+ tree nodes converging to same range
- Post-peak degradation caused by memorization overfitting, not optimization instability
- Training continued for 25 epochs after best performance (epoch 13) despite 4 LR reductions
