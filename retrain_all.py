import os
from pathlib import Path
import pandas as pd
import joblib

from pipeline import get_preprocessing_pipeline
from hybrid_model import HybridRiskPredictor
from timeline_predictor import NonLinearTimelinePredictor

BASE_DIR = Path(__file__).resolve().parent

def main():
    print("[1/5] Loading infrastructure dataset...")
    data_path = BASE_DIR / 'indian_infrastructure_projects_dataset.csv'
    df = pd.read_csv(data_path)
    
    # Feature matrix & targets
    X = df.drop(columns=[
        'delay_binary_label', 'Actual_Delay_Days', 'CRS', 'project_index',
        'delay_risk_tier', 'CRS_tier', 'section_11_notification_days', 'project_id'
    ], errors='ignore')
    
    y_binary = df['delay_binary_label'].astype(int).values
    y_crs = df['CRS'].astype(float).values
    y_days = df['section_11_notification_days'].astype(float).clip(lower=30.0, upper=730.0).values
    
    print("[2/5] Fitting leak-free preprocessing pipeline...")
    pipeline = get_preprocessing_pipeline()
    pipeline.fit(X, y_binary)
    pipeline_path = str(BASE_DIR / 'pipeline.joblib')
    joblib.dump(pipeline, pipeline_path, compress=3)
    
    print("[3/5] Transforming features...")
    X_tf = pipeline.transform(X)
    
    print("[4/5] Training Stacking Hybrid Ensemble (XGB + LGBM + CatBoost + ExtraTrees)...")
    model_params = {
        'xgb': {'n_estimators': 125, 'max_depth': 8, 'learning_rate': 0.0935},
        'lgb': {'n_estimators': 126, 'num_leaves': 44, 'learning_rate': 0.1336},
        'cat': {'iterations': 150, 'depth': 6, 'learning_rate': 0.08, 'verbose': False},
        'et':  {'n_estimators': 100, 'max_depth': 12}
    }
    
    predictor = HybridRiskPredictor(model_params=model_params)
    predictor.fit(X_tf, y_binary, y_crs, y_days)
    ensemble_path = str(BASE_DIR / 'ensemble.joblib')
    predictor.save(ensemble_path)
    
    print("[5/5] Training Non-Linear Timeline Survival Engine (RSF + DeepSurv)...")
    # Section 11 statutory notification duration
    durations = df['section_11_notification_days'].astype(float).clip(lower=30.0, upper=730.0).values
    
    timeline = NonLinearTimelinePredictor()
    timeline.fit(X_tf, y_binary, durations)
    timeline_path = str(BASE_DIR / 'timeline.joblib')
    timeline.save(timeline_path)
    
    print("\n[SUCCESS] All models retrained and serialized successfully! Ready for production API.")

if __name__ == '__main__':
    main()
