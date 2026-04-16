**Improvements from Parent**
- Multi-modal fusion: Added STRING_GNN (5.43M params, full fine-tuning) providing PPI topology embeddings
- Compact bilinear interaction head (Linear(896→896→768), reshape to [B,3,256], einsum with gene embeddings [6640,256]) replacing parent's 42M-parameter MLP head (640→2048→19920)
- Increased AIDO.Cell LoRA rank from r=32 to r=64 (4.42M trainable params vs parent 2.21M)
- Enhanced regularization: dropout increased from 0.1 to 0.25, added label smoothing 0.05
- Kept gene-position extraction and realistic multi-gene baseline from parent

**Results & Metrics (vs Parent)**
- Test F1: 0.4254 vs parent 0.4234 (+0.0020 marginal gain)
- Val F1 peak: 0.4252 at epoch 48
- Val loss stable: 0.117-0.120 range (nearly flat epoch 3-59)
- Train-val loss ratio: 1.56× (best-controlled overfitting in MCTS tree vs parent 15.5×)
- Training completed all 60 epochs (vs parent early stopped at 29)
- Performance ties STRING_GNN-only approaches (node4: F1=0.4258), far below tree best node1-2 (F1=0.4912)
- Expected threshold ≥0.44 NOT reached

**Key Issues**
- Critical: STRING_GNN cond_emb for perturbation conditioning documented but NOT implemented - all samples receive identical static PPI embeddings regardless of perturbation identity
- Bilinear interaction head too constrained vs node1-2's proven 6-layer deep residual MLP head (hidden=512, expand=4) which drives node1-2's +0.066 F1 advantage
- Missing cond_emb implementation eliminates key motivation for STRING_GNN as active conditioning signal
- AIDO.Cell contribution minimal - STRING_GNN alone achieves similar F1 without AIDO.Cell complexity
