**Improvements from Parent**
- Upgraded backbone from AIDO.Cell-3M (128-dim, 6 layers) to AIDO.Cell-100M (640-dim, 18 layers)
- Replaced learnable attention pooling with direct gene-position extraction (hidden state at perturbed gene's vocab index)
- Switched from direct QKV fine-tuning to LoRA (r=16, alpha=32) on all QKV matrices, reducing trainable parameters from ~15,700 to ~3,900 params/sample
- Changed optimizer from Muon to pure AdamW (backbone lr=1e-4, head lr=3e-4)
- Replaced standard cross-entropy with focal loss (gamma=2.0) and updated class weights to [10.91, 1.0, 29.62]
- Simplified prediction head from 2-layer (128→512→6640×3) to 2-layer (640→2048→6640×3)
- Changed scheduler from ReduceLROnPlateau to CosineAnnealingLR (T_max=100) with 10-epoch warmup

**Results & Metrics (vs Parent)**
- Test F1: 0.4096 (vs parent 0.1693) — +142% relative improvement
- Best val F1: 0.4095 at epoch 46 (vs parent 0.1696 at epoch 17)
- Training epochs: 87 (vs parent 38) before early stopping (patience=40 vs parent 20)
- Training loss: decreased from 1.17 to 0.016 (vs parent 0.76→0.34)
- Validation loss: increased monotonically from 0.90 to 5.17 (vs parent 1.40→1.50)
- Val F1 plateau: oscillated 0.405–0.410 from epoch 46 onward (vs parent peak at epoch 17 then decay)

**Key Issues**
- Severe probability calibration overfitting: val loss increased 5.74× (0.90→5.17) while val F1 remained stable
- Synthetic single-gene input bottleneck: only perturbed gene at 1.0, all 19,263 other genes at −1.0 (missing/OOD for AIDO.Cell)
- Gene-position embedding (640-dim) lacks sufficient information to predict 6,640×3 outputs
- Backbone cannot leverage learned co-expression relationships due to −1.0 masking of non-perturbed genes
- LoRA regularization prevented classification collapse (eliminated node3-1's val F1 degradation) but val F1 ceiling persists at ~0.41
- Representation capacity still insufficient relative to tree-best node1-2 (F1=0.4912)
