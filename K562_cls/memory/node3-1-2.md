**Improvements from Parent**
- Replaced focal loss with label-smoothed cross-entropy + class weights to restore stable optimization
- Replaced learnable attention-weighted fusion (256-dim weighted sum) with concatenation fusion (1280-dim input: 1024-dim AIDO.Cell + 256-dim STRING_GNN)
- Added frozen STRING_GNN embeddings (256-dim) as parallel PPI topology signal alongside AIDO.Cell-10M backbone (QKV-only, Muon)
- Implemented SGDR scheduler (T_0=15, T_mult=2) with warm restarts
- Set weight_decay=2e-2

**Results & Metrics (vs Parent)**
- Test F1: 0.4407 (vs parent 0.188, +134% relative improvement; new best for AIDO.Cell lineage)
- Val F1 peak: 0.441 at epoch 13
- Outperformed sibling by +0.008 (0.4325 → 0.4407)
- Training loss oscillated 0.92-1.20 indicating Muon instability
- Performance declined after epoch 13 through epoch 21 due to SGDR warm restart disruption at epoch 15

**Key Issues**
- Frozen direct STRING_GNN embeddings without neighborhood aggregation provide only marginal PPI topology signal compared to K=16 neighborhood attention (node1-1-1-1-1: F1=0.4846)
- SGDR warm restarts destabilized optimization on hybrid architecture rather than helping
- Muon optimizer showed training loss instability (0.92-1.20 oscillation)
- Concatenation fusion (1280-dim) may be less efficient than neighborhood attention aggregation
