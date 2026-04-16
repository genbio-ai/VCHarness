**Improvements from Parent**
- Reduced regularization: dropout 0.4→0.3, out_gene_emb weight_decay 1e-2→1e-3
- Changed schedule: SGDR micro-cycles with T_0=20, T_mult=2 (vs T_0=1200)
- Added quality-filtered SWA mechanism (top_k=15, threshold=0.500, temperature=3.0)
- Parameter count: 17.05M/22.35M trainable

**Results & Metrics (vs Parent)**
- Test F1: 0.5043 vs 0.5036 (+0.0007)
- Val best F1: 0.5041 at epoch 51
- Cycle progression confirmed: 0.43→0.48→0.48→0.49→0.50→0.504 (cycles 2-7)
- Cycle 8 degradation: overfitting (val_loss=0.75, train_loss=0.0163)
- Below grandparent node4-2 (0.5069)

**Key Issues**
- SWA mechanism failed to activate: only 2/58 epochs met threshold≥0.500, below minimum pool size of 3
- SWA threshold (0.500) too strict for this lineage
- No SWA boost realized (+0.003–0.007 expected, 0 achieved)
- Cycle 8 overfitting after 200+ epochs (val loss spike, train loss 0.0163)
