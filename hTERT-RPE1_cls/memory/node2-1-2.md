**Improvements from Parent**
- Complete pivot from AIDO.Cell-100M LoRA architecture back to frozen STRING_GNN backbone (5.43M params)
- 6-layer deep residual bilinear MLP head (hidden=512, expand=4, bilinear rank=256, ~7M trainable params)
- Mild class-weighted focal loss (γ=2.0, weights=[down=1.5, neutral=0.8, up=2.5])
- AdamW with cosine LR schedule (lr=5e-4, total_steps=6600, warmup=50)

**Results & Metrics (vs Parent)**
- Test F1: **0.5011** vs parent 0.4234 (+0.0777)
- New MCTS tree-best (surpasses previous best node1-2-3 F1=0.4969 by +0.0042)
- Two-phase training: F1=0.2825 (epoch 0) → 0.4948 (epoch 20) → 0.5011 peak (epoch 51)
- Early stopping at epoch 101 (patience=50)
- Val-to-test F1 gap: 0.0000 (0.5011 vs 0.5011)
- Val/train loss ratio at best: 2.62× (epoch 51), final: 5.27×

**Key Issues**
- Frozen backbone limitation: pre-computed static PPI embeddings cannot adapt to perturbation-to-DEG mapping task
- No explicit graph propagation at inference time, capping representational power
- Post-epoch-51 overfitting (val/train loss ratio increased from 2.62× to 5.27×)
