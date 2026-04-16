**Improvements from Parent**
- Upgraded from AIDO.Cell-3M (128-dim) to AIDO.Cell-100M with LoRA r=16 QKV fine-tuning across all 18 transformer layers
- Replaced single-query attention pooling bottleneck (19,264 genes → 128-dim) with dual-stream architecture: frozen STRING_GNN PPI embeddings (256-dim) + AIDO.Cell embeddings fused via gated projection → 640-dim combined representation
- Upgraded prediction head from simple Linear(128→512→6640×3) to 6-layer residual MLP with bilinear output interaction head (head_dim=512, 6640×512 gene embeddings)
- Changed input from unrealistic profile to realistic multi-gene knockdown: all 19,264 genes at 1.0, perturbed gene at -1.0 (AIDO.Cell "missing" token)
- Switched from standard cross-entropy to focal cross-entropy (γ=2.0) with inverse-frequency class weights [10.91, 1.0, 29.62]
- Changed optimizer from Muon (lr_muon=0.02, lr_adam=3e-4) to AdamW (backbone lr=1e-4, head lr=3e-4, weight_decay=0.01) with CosineAnnealingLR (10-epoch warmup)

**Results & Metrics (vs Parent)**
- Test F1: 0.4797 vs 0.1693 (+0.310 improvement)
- Val F1 peak: 0.4795 (epoch 35) vs 0.1696 (epoch 17)
- Early stopping at epoch 75 (patience triggered) vs epoch 38
- Training loss converged to 0.044 while validation loss diverged from 0.786 (epoch 6) to 3.004 (epoch 75) — 4× gap indicating calibration overfitting
- Test F1 (0.4797) closely matches best val F1 (0.4795), confirming good checkpoint selection generalization

**Key Issues**
- Calibration overfitting: 6-layer residual head (26M+ parameters) becomes overconfident on training data while val loss rises dramatically
- Frozen STRING_GNN embeddings (256-dim) cannot adapt to task-specific perturbation patterns, limiting fusion effectiveness
- Head overcapacity relative to trainable backbone parameters contributes to overfitting despite weight_decay=0.01
