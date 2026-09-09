import pandas as pd
import logging
import datetime
import joblib
import mlflow
import os

from pipeline import get_preprocessing_pipeline
from hybrid_model import HybridRiskPredictor
from timeline_predictor import NonLinearTimelinePredictor
from evaluate_model import evaluate_nested_cv

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_continuous_learning():
    """
    Phase 10: Retraining Pipeline
    Pulls fresh data, builds pipeline, evaluates hyperparameters via Nested CV,
    retrains models, and pushes to MLflow Model Registry.
    """
    version_tag = datetime.datetime.now().strftime("v%Y.%m.%d")
    logging.info(f"Starting Continuous Learning Retraining Pipeline ({version_tag})...")
    
    # 1. Pull Latest Data
    data_file = 'indian_infrastructure_projects_dataset.csv' if os.path.exists('indian_infrastructure_projects_dataset.csv') else 'Revolution-main/indian_infrastructure_projects_dataset.csv'
    try:
        df = pd.read_csv(data_file)
        logging.info(f"Loaded {len(df)} records for retraining from {data_file}.")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return
        
    X = df.drop(columns=['delay_binary_label', 'Actual_Delay_Days', 'CRS', 'project_index', 'delay_risk_tier', 'CRS_tier', 'section_11_notification_days', 'project_id'], errors='ignore')
    y_cls = df['delay_binary_label'].astype(int)
    y_crs = df['CRS'].astype(float)
    y_days = df['section_11_notification_days'].astype(float).clip(lower=30.0, upper=730.0)
    
    # 2. Build Pipeline
    cat_cols = ['state', 'district', 'project_type', 'terrain_type', 'sia_approval_status', 'forest_clearance_status']
    pipe = get_preprocessing_pipeline(cat_cols)
    X_tf = pipe.fit_transform(X, y_cls)
    
    # 3. Hyperparameter Optimization & Nested CV
    logging.info("Running Hyperparameter Optimization via Nested CV...")
    mlflow.set_experiment("Production_Retraining_Run")
    
    # Run a reduced trial count for speed in this script, typically higher
    outer_results = evaluate_nested_cv(X_tf, y_cls, y_crs, y_days, n_trials=5)
    
    # 4. Final Full Model Training (Using best known params or default ensemble)
    logging.info("Training final production models on full dataset...")
    final_hybrid = HybridRiskPredictor(random_state=int(datetime.datetime.now().timestamp()) % 10000)
    final_hybrid.fit(X_tf, y_cls, y_crs, y_days)
    
    final_timeline = NonLinearTimelinePredictor()
    final_timeline.fit(X_tf, y_cls, y_days)
    
    # 5. Versioning and Serialization
    save_dir = f"models/{version_tag}"
    os.makedirs(save_dir, exist_ok=True)
    
    joblib.dump(pipe, f"{save_dir}/pipeline.joblib")
    final_hybrid.save(f"{save_dir}/ensemble.joblib")
    joblib.dump(final_timeline, f"{save_dir}/timeline.joblib")
    
    logging.info(f"Successfully retrained and serialized version {version_tag} to {save_dir}")
    
    # 6. MLflow Registry (Simulated log)
    with mlflow.start_run(run_name=f"Release_{version_tag}"):
        mlflow.log_artifact(f"{save_dir}/pipeline.joblib")
        mlflow.log_artifact(f"{save_dir}/ensemble.joblib")
        mlflow.log_artifact(f"{save_dir}/timeline.joblib")
        # In a real environment, you'd register the model here using mlflow.register_model

if __name__ == "__main__":
    run_continuous_learning()
