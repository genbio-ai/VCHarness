**Improvements from Parent**
- Replaced full ESM2-650M fine-tuning with LoRA (r=16) adapters to reduce overfitting risk
- Lowered backbone learning rate to 5e-5 for LoRA parameters
- Added cond_emb perturbation injection into GNN via scatter_add_ for task-specific conditioning
- Added per-gene bias terms to MLP head for gene-specific calibration

**Results & Metrics (vs Parent)**
- Test F1: 0.4780 (+0.004 vs parent 0.4740)
- Best val F1: 0.5088 at epoch 70 (+0.0083 vs parent 0.5005 at epoch 18)
- Val-test gap: 0.0308 (worsened vs parent 0.0265)
- Training loss continued decreasing while val loss climbed from epoch 70 onward (0.065→0.100)
- Training plateaued with no val F1 improvement beyond epoch 70

**Key Issues**
- Cosine annealing T_max=140 with early stopping at epoch 90 means LR only decayed 57% of intended schedule, never entering low-LR convergence phase
- Head/GNN learning rate 1e-4 remained too aggressive throughout training, driving continuous val loss increase from epoch 13 onward
- cond_emb scatter_add_ provides only weak perturbation-specific conditioning signal to GNN
- Label smoothing (α=0.1) conflicts with focal loss's minority-class emphasis
- Training extended to 90 epochs despite best val F1 occurring at epoch 70 with no subsequent improvement
