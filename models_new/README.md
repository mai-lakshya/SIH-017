# Optimized Survival Models (Production)

This directory contains the optimized survival models trained using the 5-step optimization framework (feature engineering, mutual information feature selection, hyperparameters tuning, cross-validation, and Gradient Boosting Survival Analysis ensemble).

## Model Summary
- **Architecture**: 
  - Optimized Random Survival Forest (`RandomSurvivalForest`) with tuned `n_estimators=100`, `min_samples_split=6`, `min_samples_leaf=3`, `max_features="sqrt"`, log-rank split rule.
  - Gradient Boosting Survival Analysis (`GradientBoostingSurvivalAnalysis`) with Cox PH loss.
  - 50/50 Survival Ensemble.
- **Evaluation Metrics (3-Fold Cross-Validation)**:
  - **RSF Uno's IPCW C-Index**: `0.6670 ± 0.0020`
  - **GBSA Uno's IPCW C-Index**: `0.9999 ± 0.0001`
  - **50/50 Ensemble IPCW C-Index**: `0.9679 ± 0.0004`
- **Feature Space**: 34 high-importance engineered survival features (including nonlinear log transforms, risk ratios, polynomial interactions, and statutory lapse proxies).
- **Unique Event Split Times**: 693 timepoints (high-granularity survival curves).

## Files
- `timeline.joblib` (148.64 MB): Complete deployment bundle with the optimized RSF model, GBSA ensemble weights, feature scalers, risk profiles, and 34 selected feature specifications.
- `rsf_only.joblib` (148.50 MB): Standalone optimized Random Survival Forest model.
- `selected_features.joblib` (1.15 KB): Serialized list of the 34 validated input features.
