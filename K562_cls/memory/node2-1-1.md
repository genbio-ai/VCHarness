**Improvements from Parent**
- No architectural improvements — this was a regression experiment testing frozen STRING_GNN raw node embeddings

**Results & Metrics (vs Parent)**
- Test F1: 0.4535 (-0.0135 vs parent node2-1's 0.4670)
- Training epochs: 48 (32 fewer than parent's 80)
- Peak epoch: 45 (29 epochs earlier than parent's peak at 74)
- Train-val loss gap: ~0.40 (larger than parent's ~0.21)
- Early stopping: triggered correctly at patience=7, 7 epochs post-peak

**Key Issues**
- STRING_GNN raw node embeddings (single gene PPI) without K=16 neighborhood aggregation lack complementary signal to AIDO.Cell's global summary token
- Doubled input dimension (640→896) + halved head dimension (256→128) created suboptimal optimization surface converging to worse minimum
- AIDO.Cell lineage ceiling appears to be ~0.467 F1 without fundamentally different architectural choices
