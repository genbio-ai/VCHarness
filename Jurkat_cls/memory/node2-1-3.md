**Improvements from Parent**
- Architecture: Changed from dual-pooling MLP head to 4-token cross-attention transformer fusion (AIDO.Cell-10M + LoRA r=4 all layers + Symbol CNN + frozen STRING GNN → 3-layer pre-norm TransformerEncoder → head)
- Loss: Changed from standard loss to focal loss with class_weights=[6,1,12] + label_smoothing=0.05
- Regularization: Added manifold mixup (alpha=0.3), increased weight_decay from 0.01 to 0.10
- Optimizer: Changed from OneCycleLR to AdamW with separate learning rates (backbone lr=2e-4, others 6e-4)

**Results & Metrics (vs Parent)**
- Test F1: 0.4633 vs parent 0.4101 (+0.053)
- Train-val loss gap: ~1.5-2× vs parent 27× overfitting
- Performance: +0.018 over best sibling, −0.0135 below tree-best (0.4768)

**Key Issues**
- GPU constraint: 2 GPUs with accumulate_grad_batches=4 vs tree-best's 8-GPU configuration, likely affecting gradient quality and batch statistics
- FlashAttention: Pre-norm transformer fusion warning indicates optimization disabled, potentially slowing convergence
