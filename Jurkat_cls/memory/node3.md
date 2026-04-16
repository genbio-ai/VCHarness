**Technical Implementation**
- Model: AIDO.Protein-16B with LoRA fine-tuning
- LoRA configuration: r=8, applied to last 4 layers
- Input representation: protein amino acid sequences of knocked-out genes
- Training duration: 28 epochs
- Learning rate schedule: reduction at epoch 18

**Results & Metrics**
- Test F1: 0.0498
- Validation F1 progression: 0.0128 → 0.0498 (improvement through epoch 12)
- Performance plateau: no improvement after epoch 12, no recovery following LR reduction
- Relative performance: ~8× worse than best node (node1 F1: 0.39)
- Training set size: 1,500 knockout examples

**Key Issues**
- Protein sequences encode molecular function but lack regulatory network position information
- Protein language model incapable of learning meaningful differential expression patterns from knockout data
- Protein sequence representation fundamentally mismatched for DEG prediction task
- Requires gene identity encoding capturing functional and regulatory relationships (e.g., gene symbol embeddings, PPI graph embeddings, genomic regulatory context)
