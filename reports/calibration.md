# Temperature Calibration

- Checkpoint: `runs/p8-gated-kl1/model.pt`
- Protocol: `bias_matched`
- Split: `val_matched` (904 images)
- Temperature: `0.162786`
- NLL before/after: `0.024487` / `0.020112`
- ROC AUC: `0.999985`
- Target FPR: `1.00%`
- Threshold: `0.427470`
- Achieved FPR: `0.88%`
- Accuracy at threshold: `0.995575`

The threshold is fitted on calibrated clean/TTA scores and must be used only with the checkpoint, protocol, crop grid, and TTA settings recorded in the JSON artifact.
