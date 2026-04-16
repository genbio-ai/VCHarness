**Improvements from Parent**
- LoRA rank increased from r=16 to r=32 (Q/K/V layers), trainable backbone params expanded to 2.21M
- Representation enriched from single gene-position vector (640-dim) to 1280-dim concatenation: gene-position embedding + mean-pool over all 19,264 genes from weighted fusion of last 6 layers
- Decoder upgraded from single-layer linear (640→19920) to bilinear prediction head (dim=512, ~5.57M params)
- Loss function changed from class-weighted CE to focal loss (γ=2.0) + class weights [2.0, 1.0, 4.0] + label smoothing (0.05)
- LR schedule switched from ReduceLROnPlateau to cosine (T_max=4500 steps ≈ 100 epochs)

**Results & Metrics (vs Parent)**
- Test F1: 0.4391 vs 0.3445 (+0.0946), best in node2 lineage (surpasses node2-1: 0.4234, node2-2: 0.4102)
- Best val F1: 0.4388 at epoch 19 vs 0.3446 at epoch 67
- Training duration: 50 epochs (early stopping patience=30) vs 83 epochs
- Training loss: 0.635→0.094 (monotonic decrease)
- Val loss: minimum 0.262 at epoch 3, increased to 0.367 at epoch 49 (overfitting onset)
- Backbone LR at termination: 2.6e-5 (52% of peak 5e-5), cosine schedule only ~50% complete

**Key Issues**
- Cosine LR schedule misaligned with early stopping: T_max=4500 set for ~100 epochs but training stopped at epoch 50, never reaching low-LR fine-tuning phase
- Overfitting with 7.8M trainable params vs 1,416 training samples; val F1 plateaued from epoch 19 onward
- Mean-pool signal dilution suspected: all 19,263 non-perturbed genes normalized to identical expression=1.0, making mean-pool nearly constant across samples
- 5% F1 gap to tree best node1-2 (STRING_GNN: 0.4912) suggests AIDO.Cell expression-based representations may have lower ceiling than PPI-topology-based approaches
