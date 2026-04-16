**Technical Implementation**
- AIDO.Protein-16B (16B MoE) with LoRA fine-tuning (r=16, α=32)
- Mean pooling → 2-layer MLP → [B,3,6640] logits
- Focal Loss with γ=2.0
- Aggressive inverse-frequency class weights

**Results & Metrics**
- Test F1: 0.0378
- Val F1: 0.0396
- Training loss collapsed from 7e-4 to ~1e-11 within first epoch
- Val loss increased monotonically

**Key Issues**
- Catastrophic model collapse: complete memorization without generalization
- Extreme class imbalance in weights (class 2 weight ≈ 38× class 0)
- Focal loss + class weights created feedback loop saturating training loss and killing gradient signal
- Near-random predictions for minority classes
