**Improvements from Parent**
- Replaced AIDO.Protein-16M protein sequence encoder with AIDO.Cell-10M single-cell encoder (LoRA r=4, all 8 layers)
- Added 3-branch character-level symbol CNN (64-dim) for gene symbol embeddings
- Introduced frozen STRING GNN PPI topology embeddings (256-dim) to capture gene-gene interaction network position
- Implemented 4-source feature fusion (832-dim total: 512 + 64 + 256) with 384-dim MLP head
- Switched loss to Focal Loss (γ=2.0, class weights [5.0, 1.0, 10.0]) to handle class imbalance
- Replaced CosineAnnealingLR with ReduceLROnPlateau (patience=8, monitor=val_f1)

**Results & Metrics (vs Parent)**
- Test F1: 0.462 vs parent 0.0498 (9.3× improvement)
- New best in MCTS tree, surpassing node2-2-1 (0.447) and node3-1 (0.243)
- Best checkpoint at epoch 18: test F1 = val F1 = 0.462
- Frozen PPI embeddings provided additional biological signal breaking previous ~0.45 ceiling

**Key Issues**
- Severe overfitting after epoch 18: training loss 0.118→0.065 while val_loss 0.587→0.845
- 40 wasted epochs with no improvement after overfitting onset
- ReduceLROnPlateau scheduler never fired because val_f1 oscillated ±0.005 within patience window (monitoring val_f1 instead of val_loss)
