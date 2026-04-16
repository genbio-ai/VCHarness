**Improvements from Parent**
- Backbone: AIDO.Cell-10M → scFoundation-100M (768-dim)
- Input encoding: 19,264-token uniform → sparse context (1,649 perturbation panel genes at 100.0, knocked-out gene at 0.0)
- Added 128-dim gene symbol embedding
- LoRA targeting: all 8 layers (AIDO.Cell) → out_proj/linear2 projections only (scFoundation 12 layers)
- LoRA dropout: 0.25 → 0.3 (r=8 unchanged)

**Results & Metrics (vs Parent)**
- Test F1: 0.3809 vs parent 0.4101 (Δ = −0.0292)
- Val_f1: 0.349–0.381 with single peak at epoch 8, never recovered
- Val_loss: 0.362 → 0.506 (steady climb)
- Train_loss: decreased to 0.154 (monotonic)
- Train-val gap: 3.5×
- Early stopping: epoch 23 (15 consecutive epochs with no val_f1 improvement)

**Key Issues**
- Sparse context (1,649 near-identical tokens) provides less positional diversity than parent's 19,264-token uniform encoding
- LoRA frozen QKV attention weights prevents 12-layer transformer from adapting attention patterns to DEG task
- Frozen attention cannot compensate for information-impoverished sparse input
