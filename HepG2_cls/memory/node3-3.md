**Improvements from Parent**
- Replaced AIDO.Cell-100M LoRA synthetic single-gene expression encoding with frozen STRING_GNN 256-dim PPI embeddings
- Changed from 3-layer MLP (h=6640) to 4-block Pre-Norm residual MLP (h=512, inner=1024, dropout=0.40) with per-gene bias
- Switched optimizer from Adam (lr=1e-4) to AdamW (lr=3e-4) with 5-epoch warmup and RLROP (patience=10)

**Results & Metrics (vs Parent)**
- Test F1: 0.387 vs 0.157 (+0.230 absolute improvement)
- Training: 182 epochs with 3 RLROP halvings at epochs 70/164/175
- Final train loss: 0.947 vs parent 0.281 (degraded)
- Performance gap: −0.088 below tree ceiling (F1=0.474–0.478)
- Sibling comparison: +0.010 vs node3-2 (F1=0.377)

**Key Issues**
- Severe underfitting: train loss 0.947 is 79× worse than node1-1-1 (0.012) and 4.5× worse than node1-3-2 (0.211)
- Growing train-val loss gap: 0.085→0.221, indicating simultaneous underfitting and overfitting
- Premature first RLROP halving at epoch 70 interrupted 4-block model's steep improvement phase
- Architecture-mismatch: hidden_dim=512 with 4 blocks + dropout=0.40 + wd=5e-4 is fundamentally incompatible with 1,273 samples
- Tree-wide evidence confirms h=384 with Muon is optimal for STRING-only; no 4-block variant achieved F1 > 0.43
