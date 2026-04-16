**Improvements from Parent**
- LoRA expanded from r=8 (last-4 layers) to r=4 (all 8 transformer layers)
- Added stronger class weights: [6.0, 1.0, 12.0]
- Added label smoothing=0.05
- Added SWA with swa_lrs=5e-5, swa_epoch_start=20
- Increased ReduceLROnPlateau patience from 5 to 12
- Increased early stopping patience from default to 20

**Results & Metrics (vs Parent)**
- Test F1=0.4555 vs parent 0.4513 (+0.004 gain)
- Best val_f1=0.455 at epoch 15 vs parent 0.4513 at epoch 13
- Ran 35 epochs vs parent 38 epochs
- SWA replaced ReduceLROnPlateau at epoch 19 (after best epoch)
- Val_loss exploded from 0.631 to 0.815 during SWA phase (+0.184)
- Tree-best comparison: −0.007 below node3-2 (0.4622)

**Key Issues**
- SWA started too late (4 epochs after best epoch)
- SWA learning rate too high (swa_lrs=5e-5)
- SWA disabled ReduceLROnPlateau's beneficial LR adaptation post-peak
- SWA averaged weights from degrading regime rather than converged regime
- Val_loss deterioration during SWA phase indicates unstable averaging
