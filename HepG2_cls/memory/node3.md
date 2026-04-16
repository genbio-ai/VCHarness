**Technical Implementation**
- AIDO.Cell-100M with LoRA (r=16, Q/K/V)
- Synthetic single-cell expression encoding: perturbed gene at expression 1.0, all other 19,263 genes masked at -1.0
- Mean-pooling over 19,264 gene positions
- 3-layer MLP head → `[B, 3, 6640]`
- Weighted cross-entropy loss with label smoothing
- 1,273 training samples

**Results & Metrics**
- Test F1: 0.157
- Validation F1: 0.446 (constant across all 18 epochs)
- Training loss: 0.519 → 0.281 over 18 epochs
- 0.446 equals naive baseline of predicting all genes as neutral (92.8% class-0 distribution)

**Key Issues**
- Synthetic single-gene expression input is radically out-of-distribution relative to AIDO.Cell's pre-training on diverse transcriptomes
- Model cannot extract meaningful biological signal from a cell with only 1 of 19,264 genes unmasked
- Mean-pooling over 19,264 positions washes out remaining signal
- Model found degenerate solution that minimizes weighted CE without learning gene-response discrimination
- Significantly underperforms vs. node1-1 (STRING_GNN PPI embeddings, test F1=0.472)
