**Improvements from Parent**
- Replaced perturbation-conditioned dynamic STRING_GNN with pre-computed static STRING embeddings (256-dim), eliminating the cond_proj(256→256) and batch-wise conditioning complexity
- Added character-level 3-branch CNN on Ensembl IDs as third input modality (64-dim)
- Changed from frozen AIDO.Cell-10M to LoRA fine-tuning (r=4, all 8 Q/K/V layers)
- Reduced head width from 768→128→19920 to 768→384→19920 (single intermediate layer)
- Adjusted focal loss hyperparameters: γ=2.5→2.0, class_weights=[3,1,7]→[5.0,1.0,10.0], label_smoothing=0.12→0.05
- Replaced CosineAnnealingWarmRestarts with ReduceLROnPlateau (factor=0.5)

**Results & Metrics (vs Parent)**
- Test F1: 0.4300 (vs parent 0.4490, delta -0.019)
- Best val_f1: 0.4301 at epoch 38 (vs parent 0.4354 at epoch 71)
- Val-test gap: 0.0001 (vs parent -0.014, test below val)
- Training loss: 0.132→0.105 at peak (−21%), continued dropping post-peak
- Early stopping triggered after 63 total epochs (25 epochs past optimum)
- ReduceLROnPlateau fired twice (E47, E56) with degradation after each
- Val_f1 regression from 0.4301→0.422 during wasteful post-peak epochs

**Key Issues**
- Performance regression of -0.019 vs parent (-0.032 vs tree-best node3-2 F1=0.462) despite identical fusion architecture
- Early stopping patience=25 too permissive, allowed 25 wasteful epochs after optimal checkpoint at E38
- ReduceLROnPlateau LR reductions counterproductive: both firings (E47, E56, factor=0.5) caused val_f1 degradation rather than recovery
- Post-peak overfitting: train_loss continued dropping 21% while val_f1 regressed 0.008
- Training dynamics mismanagement identified as primary bottleneck (not architecture flaw) given near-zero val-test gap confirms good generalization
