**Improvements from Parent**
- Changed to unified learning rate (5e-5 for all components) instead of split LR (LoRA=5e-5, head/GNN=1e-4)
- Reduced cosine T_max from 140 to 60 epochs
- Removed label smoothing (previously 0.05)

**Results & Metrics (vs Parent)**
- Val F1: 0.5225 at epoch 61 (+0.0137 vs parent's 0.5088)
- Test F1: 0.4642 (-0.0138 vs parent's 0.4780)
- Val-test gap: 0.0583 (+0.0275 wider vs parent's 0.0308)
- test_score.txt: 0.2198 (incorrect due to eval.py checkpoint loading bug)

**Key Issues**
- Overfitting: val-test gap nearly doubled from parent
- Unified LR disrupted STRING_GNN pretrained weights
- Removing label smoothing caused overconfident minority-class predictions
- Premature cosine schedule minimum (T_max=60 too aggressive)
- eval.py checkpoint loading bug strips "esm2." prefix, silently dropping LoRA weights
