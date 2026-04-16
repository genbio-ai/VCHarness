**Improvements from Parent**
- Added learned scalar gating (1-dim gate from 2560-dim concat) to fuse AIDO.Protein-16B and STRING_GNN embeddings instead of simple concatenation
- Increased LoRA rank from r=16 to r=32 on last 12 of 36 layers
- Replaced single-layer mean pooling with 4-layer weighted hidden state aggregation with attention pooling
- Expanded head from 2-layer MLP to 3-layer bottleneck architecture (2304→512→256→19920) with per-gene bias
- Switched from Focal Loss (γ=2.0) to sqrt-inverse weighted CE with label_smoothing=0.05
- Changed learning rate to differential schedule (backbone_lr=5e-5, head_lr=1.5e-4)
- Added ReduceLROnPlateau scheduler (patience=8)

**Results & Metrics (vs Parent)**
- Test F1: 0.4049 (identical to parent node2-1, no improvement)
- Val F1: Improved from 0.4048 (epoch 0) to 0.4597 (epoch 15), but gains did not transfer to test set
- Performance ceiling: 0.069 below tree best (node1-1-1, node4: F1=0.474)
- ReduceLROnPlateau never triggered (lr remained at 5e-5 for all 21 epochs)

**Key Issues**
- Complete generalization failure: validation F1 gains disappeared at test time, indicating learning of spurious validation-specific patterns
- AIDO.Protein-16B protein sequence features show no biologically relevant signal for HepG2 perturbation response across 5 node2 lineage experiments
- All architectural variants (concat, gating, restricted LoRA, multi-layer pooling, longer sequences) converge to same F1 ≈ 0.40 ceiling
- Increased complexity (gating mechanism, multi-layer pooling, restricted LoRA) provided zero benefit
- Core bottleneck: protein sequence representation alone insufficient for perturbation prediction task
