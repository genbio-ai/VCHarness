**Improvements from Parent**
- Fixed critical AIDO.Cell model bug: switched from 3M (256-dim features) to 10M (512-dim features)
- Replaced concat+MLP fusion with cross-attention fusion using 4-token TransformerEncoder
- Changed LoRA configuration: r=8 last-4-layers → r=4 all-8-layers
- Updated focal loss parameters: [3.0,1.0,5.0] γ=2.0 → [6,1,12]
- Added manifold mixup with alpha=0.3
- Increased weight_decay: 0.05 → 0.10
- Adjusted differential LR: 1e-4/3e-4 → 2e-4/6e-4

**Results & Metrics (vs Parent)**
- Test F1: 0.4678 vs parent 0.4171 (+0.051 improvement)
- Val F1 progression: 0.368→0.468 across epochs 0-47
- Plateau at val_f1~0.466 from epoch 33 onward
- Two LR reductions (2e-4→5e-5) during plateau phase
- Early stopping triggered at epoch 67 after 20 epochs without improvement
- Zero val-test gap confirming no overfitting

**Key Issues**
- Top-3 checkpoint averaging disabled due to DDP multi-GPU constraint (~+0.003 potential gain missed)
- Precomputed STRING embeddings from pretrained (not fine-tuned) GNN limit fusion quality
- Still -0.009 below tree-best performance (0.4768)
