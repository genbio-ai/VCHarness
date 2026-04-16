**Improvements from Parent**
- Switched from AIDO.Cell-10M with single-gene sparse profile to AIDO.Cell-100M with hybrid dual-stream representation (Stream A: perturbed gene position embedding, Stream B: mean of two summary tokens)
- Introduced LayerNorm + Linear(1280, 640) + GELU fusion before bilinear head (rank=256)
- Reduced LoRA rank from r=8 to r=16 on larger backbone
- Changed from inverse-frequency to fixed class weights [2.0, 1.0, 4.0]
- Lowered differential learning rates (backbone: 1e-4 → 5e-5, head: 5e-4 → 3e-4)

**Results & Metrics (vs Parent)**
- Test F1: **0.4411** (vs parent 0.3089): **+0.1322 improvement**
- Best val F1: 0.4410 at epoch 25 (vs parent 0.3089 at epoch 30)
- Test-val generalization gap: ~0.001 (excellent checkpoint selection)
- Training phases: rapid ramp-up (epochs 0-8, val_f1: 0.36→0.41), improvement (epochs 9-25, val_f1: 0.40→0.44), plateau (epochs 26-40, mean val_f1=0.4323, std=0.0050)
- Early stopping at epoch 40 (patience=15), final train-val loss gap: 0.1661 (mild overfitting)
- Outperformed sibling node1-1-1 (mean-pool) by +0.0742 and node4 by +0.0153

**Key Issues**
- AIDO.Cell backbone representation of artificial multi-gene baseline ({perturbed_gene: 1.0}) provides limited cross-perturbation diversity for summary tokens
- Gene-position embedding encodes perturbation context from synthetic expression profile rather than real transcriptional data
- Performance ceiling limited by unrealistic input representation despite superior architecture to single-gene OOD input
- Cosine LR schedule aligned to 80 epochs but training stopped at epoch 40; early-stopping patience potentially conservative for final plateau phase
