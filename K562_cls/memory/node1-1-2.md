**Improvements from Parent**
- Added frozen ESM2-650M protein sequence embeddings (3840-dim) as second backbone alongside frozen STRING_GNN PPI embeddings (256-dim)
- Fused dual backbones to 4096-dim representation (concatenation)
- Simplified projection to 2-layer MLP (4096→512→256) vs parent architecture
- Reduced dropout from 0.4 to 0.25
- Changed loss from class-weighted CE with label smoothing to asymmetric focal loss (γ_neutral=3.0, γ_deg=1.0)
- Replaced cosine annealing with linear warmup + CosineAnnealingWarmRestarts (T_0=100, T_mult=2)
- Changed weight decay from parent setting to 1e-2

**Results & Metrics (vs Parent)**
- Test F1: 0.4585 vs parent 0.4527 (+0.006)
- Test F1: 0.4585 vs sibling node1-1-1 0.4439 (+0.015)
- Test F1 0.4585 = best in entire search tree
- Val F1 peak: 0.4585 at epoch 95
- Val F1 final: 0.456 at epoch 116 (mild decline from peak)
- Trained 116 epochs (reached max_epochs limit, early stopping did not trigger)
- Train-val loss gap: ~0.067 (mild overfitting, better than sibling's deep residual architecture)
- Val loss stable throughout late training

**Key Issues**
- Both backbones (STRING_GNN, ESM2) encode static, non-perturbation-aware representations
- Model must learn protein-to-transcriptional-response mapping from limited 1,388 training samples
- Performance ceiling around 0.46 F1
- Val F1 declined from epoch 95 to 116 (mild overfitting pattern)
- Training reached max_epochs before early stopping could intervene
- Bilinear head treats each gene independently without gene-gene regulatory relationships
