**Improvements from Parent**
- Frozen STRING_GNN backbone with pre-computed 256-dim PPI embeddings
- Neighborhood attention module (K=16, attn_dim=64)
- GenePriorBias initialization (log-class-frequencies, gradient-zeroed for 50 epochs)
- 20-epoch linear warmup + cosine annealing
- Dropout reduced from 0.4 to 0.35, weight_decay=2e-2

**Results & Metrics (vs Parent)**
- Test F1=0.4093 vs 0.4527 (regression −0.043, worst in node1 lineage)
- 16-epoch cold-start failure: val/f1 stuck at 0.353 (majority baseline) while train loss decreased 1.48→0.94
- Model escaped neutral-prediction trap only after LR warmup completed (epoch 20+)
- Best checkpoint at epoch 46 (val/f1=0.4093), early stopping at epoch 54

**Key Issues**
- Frozen STRING_GNN backbone provides no adaptive signal
- GenePriorBias initialization [-3.15, -0.078, -3.44] created 25× neutral preference locked by 50-epoch warmup
- 20-epoch LR warmup prevented early escape from bias trap
- Neighborhood attention ineffective on frozen suboptimal STRING PPI embeddings
