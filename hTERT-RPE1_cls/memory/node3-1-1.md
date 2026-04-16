**Improvements from Parent**
- Switched from AIDO.Cell-10M to AIDO.Cell-100M backbone (hidden_size=640, 18 layers vs. 256, 8 layers)
- Replaced QKV-only direct fine-tuning (22.26M params, 70% trainable) with LoRA adapters on all QKV matrices (~2.7M params, rank=8, alpha=16)
- Changed optimizer from Muon to pure AdamW (lr_backbone=1e-4, lr_head=3e-4, weight_decay=0.01)
- Switched from weighted cross-entropy to focal loss (gamma=2.0, label_smoothing=0.05, same class_weights [10.91, 1.0, 29.62])
- Widened prediction head from Linear(256→1024→6640×3) to Linear(640→2048→6640×3) with dropout=0.2
- Inherited gene-position extraction architecture from Node 3-1

**Results & Metrics (vs Parent)**
- Test F1: 0.4096 vs. parent 0.3853 (+0.0243 absolute improvement)
- Best val F1: 0.4094 at epoch 12 (parent: 0.3850 at epoch 69)
- Training duration: 43 epochs before early stopping vs. parent 95 epochs (patience 30 vs. 25)
- Val-to-test generalization gap: ≈0.0002 (near-perfect checkpoint selection, similar to parent)
- Training dynamics: rapid improvement phase (epochs 0–12, val F1: 0.3095→0.4094), then 30-epoch plateau (val F1: 0.3804–0.4085, mean=0.4020, std=0.006)
- Final training loss: 0.0172; final val loss: 1.4698 (val-train loss gap ~1.45)
- Performance gap vs. Node 4 (STRING_GNN): 0.4258 – 0.4096 = 0.0162 F1

**Key Issues**
- Early plateau at val F1 ≈ 0.41 after only 12 epochs, followed by 30-epoch stagnation despite continued training
- LoRA rank=8 constrains backbone updates to 8-dimensional subspace of 640×640 weight space (2.7M trainable vs. parent's 22.26M)
- Sparse input encoding {perturbed_gene: 1.0, all_others: -1.0} is out-of-distribution relative to AIDO.Cell-100M's pre-training regime
- Val loss divergence (0.7868→1.4698) while training loss collapsed to 0.0172, indicating calibration deterioration despite stable argmax predictions
- Falls short of STRING_GNN (Node 4) by 0.0162 F1, which encodes protein-protein interaction topology absent from transcriptomic backbone
