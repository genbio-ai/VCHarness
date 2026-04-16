**Improvements from Parent**
- Replaced AIDO.Cell-100M synthetic expression encoding with STRING_GNN frozen PPI graph embeddings (256-dim)
- Changed from mean-pooling degenerate solution to 3-block pre-norm residual MLP (512→1024→512) with dropout 0.35
- Added additive per-gene bias parameters (6,640 genes × 3 classes)
- Fixed learning dynamics: changed from frozen validation F1=0.446 to gradual monotonic improvement (0.235→0.378 over 172 epochs)
- Switched optimizer from Adam to AdamW (lr=3e-4, wd=5e-4) with CosineAnnealingLR (T_max=200)

**Results & Metrics (vs Parent)**
- Test F1: **0.3773** (vs parent 0.157) — +0.220 absolute improvement
- Training epochs: 172 (vs parent 18) — 9.6× longer training
- Validation F1 progression: 0.235→0.378 (vs parent frozen at 0.446 naive baseline)
- Training loss: successfully decreased (vs parent 0.519→0.281 with no learning)
- Comparison to best tree node: **-0.097 regression** from node1-1-1 (F1=0.474)
- Early stopping: not triggered despite patience=25 (training reached epoch 172)

**Key Issues**
- **Learning rate schedule bottleneck**: CosineAnnealingLR with T_max=200 decays LR too slowly (lr ≈ 5×10⁻⁵ at best epoch 145); node1-1-1's ReduceLROnPlateau (factor=0.5) halved LR at epochs 52-53 and 61, enabling escape from suboptimal local minimum
- **Performance ceiling**: Current architecture (STRING_GNN + 3-block + per-gene bias) underperforms best tree node by 0.097 F1 despite correct implementation
- **Code verified correct**: All components validated (STRING_GNN embeddings, class weights, per-gene bias, loss computation, distributed evaluation) — performance gap attributable to training dynamics, not implementation bugs
- **Most actionable fix**: Replace CosineAnneiningLR with ReduceLROnPlateau (factor=0.5, patience=8-10) while keeping proven STRING_GNN + 3-block + per-gene bias architecture
