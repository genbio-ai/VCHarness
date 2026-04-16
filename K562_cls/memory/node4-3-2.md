**Improvements from Parent**
- Removed SWA averaging (fixed parent's catastrophic 68-checkpoint averaging that degraded test F1)
- Fixed checkpoint metric naming (avoided sibling's F1=0.3532 evaluation bug)
- Added NeighborhoodAttentionModule (K=16 neighborhoods, attn_dim=64, +164K params)
- Added GatedFusion for combining scFoundation and STRING embeddings
- Added two-stage GenePriorBias (warmup=50 epochs, scale=0.5)
- Added gene-frequency-weighted loss (boost=2.0 for top-2000 HVG genes)

**Results & Metrics (vs Parent)**
- Test F1: 0.4650 (vs parent 0.4585, +0.0065 recovery)
- Best checkpoint: epoch 67 with zero val-test gap
- Oscillatory plateau through epoch 92 before EarlyStopping fired
- Premature convergence at epoch 67 (vs comparable nodes peaking at epochs 167 and 222)
- Gap to lineage best (node4-2-1-2: 0.4893): -0.0243
- Sharp 0.014 val F1 drop at epoch 68 due to optimization instability

**Key Issues**
- NeighborhoodAttentionModule creates redundancy: scFoundation already provides perturbation-specific context, making neighborhood attention unnecessary in the fusion architecture
- Gene-frequency-weighted loss combined with NeighborhoodAttention causes optimization instability (sharp F1 drop at epoch 68)
- Early convergence (epoch 67) prevented reaching sharper optimum achieved by node4-2-1-2 without neighborhood attention
- min_lr_ratio=0.05 too high for stable convergence in this architecture
