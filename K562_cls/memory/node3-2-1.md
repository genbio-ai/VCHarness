**Improvements from Parent**
- Expanded FFN fine-tuning from last 2 to last 4 transformer layers in AIDO.Cell-10M backbone
- Increased Muon learning rate from 0.02 to 0.03
- Increased weight decay from 2e-2 to 3e-2
- Reduced warmup from 5 to 3 epochs
- Extended max_epochs budget to 200 with cosine LR schedule

**Results & Metrics (vs Parent)**
- Test F1: 0.4142 vs 0.4295 (regression of -0.0153 F1, -3.6%)
- Best val F1: 0.4141 at epoch 19 vs 0.4294 at epoch 24
- Training ran 30 epochs before early stopping
- Backbone trainable params: ~3.14M (QKV + 4 FFN layers) vs parent 2.56M
- F1 fluctuated 0.34-0.41 range vs parent's stable 0.35-0.43

**Key Issues**
- 33% increase in backbone trainable parameters created excess capacity causing overfitting on 1,388 training samples
- 50% LR increase destabilized optimization trajectory (higher F1 fluctuation vs parent)
- Cosine schedule provided minimal benefit — warmup consumed ~3% of 330-step budget with virtually no decay phase before early stopping
- Model peaked earlier (epoch 19 vs 24) with worse absolute performance due to combined excess capacity, aggressive LR, and insufficient regularization
