**Improvements from Parent**
- Replaced random gene-symbol embeddings with pretrained AIDO.Cell-10M backbone + LoRA (r=8, alpha=16)
- Switched from weighted cross-entropy to focal loss (gamma=2.0) with inverse-frequency class weights
- Changed optimization from ReduceLROnPlateau to cosine annealing LR schedule with warmup (backbone lr=1e-4, head lr=5e-4)
- Replaced MLP head with bilinear interaction head: pert_emb [B,256] → [B,3,128] × out_gene_emb[6640,128]^T → logits [B,3,6640]
- Used gene-specific hidden state extraction from AIDO.Cell position indexing

**Results & Metrics (vs Parent)**
- Test F1: **0.3089** vs parent 0.3762 (Δ = −0.0673, significant regression)
- Best val F1: 0.3089 at epoch 30 (vs parent 0.3763 at epoch 60)
- Train F1 ~ val F1 (negligible generalization gap on F1 metric)
- Train loss: 2.755 → 1.442 over 45 epochs
- Val loss: min 1.801 at epoch 11 → 1.897 at epoch 45 (monotonic increase after epoch 11)
- Train-val loss gap: 0.455 at epoch 45 (moderate overfitting)
- Val F1 variance: std=0.023, with frequent 0.02–0.05 swings per epoch (unstable optimization)

**Key Issues**
- **Critical OOD input representation**: Single-gene expression profile (1 gene at 1.0, rest missing) prevents AIDO.Cell attention from computing meaningful co-expression context, yielding uninformative gene embeddings
- Severe performance regression (−0.0673 F1) compared to random-embedding MLP baseline, indicating pre-training benefit is negated by inappropriate input format
- Unstable optimization with high val F1 variance and val loss monotonic increase after epoch 11
- Bilinear head architecture increases parameters compared to parent but fails to leverage pretrained embeddings effectively
