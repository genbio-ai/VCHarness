**Improvements from Parent**
- Replaced focal loss (gamma=2.0) with label-smoothed cross-entropy (ε=0.1) + sqrt-inverse-frequency class weights
- Replaced learnable attention-weighted layer fusion with STRING_GNN K=16 neighborhood attention (attn_dim=64, center-context gating, 98K trainable params) fused via concatenation (1280-dim)
- Reduced MLP head dimension from 512 to 256 (dropout=0.4)
- Maintained AIDO.Cell-10M QKV-only fine-tuning with Muon optimizer and CosineAnnealingLR (T_max=80, no restarts)

**Results & Metrics (vs Parent)**
- Val F1=0.437 at epoch 27 (vs parent 0.414, +0.023 improvement)
- Test F1=0.188 (identical to parent 0.188, no improvement)
- Val-test gap of -0.249 (exactly matches parent's failure mode)
- Sibling comparison: 0.188 vs 0.4325 and 0.4407 (significantly worse)
- Test predictions: 99.95% neutral argmax, mean class 1 probability=0.71 vs expected 0.925

**Key Issues**
- Catastrophic test failure: test F1=0.188 despite competitive val F1=0.437
- Learnable STRING neighborhood attention on frozen STRING embeddings creates training-specific patterns that fail to generalize
- Under-parameterized head (256-dim) combined with neighborhood attention produces degenerate predictions biased 99.95% toward neutral
- Validation metric is unreliable due to 92.5% neutral class distribution masking failure to learn discriminative DEG patterns
- Val-test gap exactly matches parent node3-1's failure, indicating architectural approach is fundamentally flawed
