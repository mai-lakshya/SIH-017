# Baseline Models (Legacy)

This directory contains the original baseline survival models before the 5-step optimization pipeline.

## Model Summary
- **Primary Model**: Baseline Random Survival Forest (`RandomSurvivalForest`)
- **Evaluation Metric**: Uno's IPCW C-Index: ~0.782 benchmark / 0.6721 cross-validated baseline
- **Feature Space**: 28 baseline tabular risk features (without interaction/log transformations)
- **Unique Event Split Times**: 168 timepoints

## Files
- `timeline.joblib` (1.42 MB): Original pipeline dictionary containing the baseline RSF model and timeline artifacts.
- `rsf_only.joblib` (30.20 MB): Standalone baseline Random Survival Forest estimator.
