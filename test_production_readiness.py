import pytest
import joblib
import pandas as pd
import numpy as np
import math
import time
import psutil
import os
import json
from datetime import datetime

# Import system modules
from pipeline import get_preprocessing_pipeline
from hybrid_model import HybridRiskPredictor
from timeline_predictor import NonLinearTimelinePredictor
from explainer import DualParadigmExplainer

np.random.seed(42)

@pytest.fixture(scope="session")
def artifacts():
    """Load all ML artifacts once per session"""
    try:
        pipeline = joblib.load('pipeline.joblib')
        predictor = HybridRiskPredictor.load('ensemble.joblib')
        timeline = joblib.load('timeline.joblib')
        
        # Load sample data to initialize explainer
        df = pd.read_csv('indian_infrastructure_projects_dataset.csv')
        X = df.drop(columns=['delay_binary_label', 'Actual_Delay_Days', 'CRS', 'project_index'], errors='ignore')
        X_tf = pipeline.transform(X.head(200))
        explainer = DualParadigmExplainer(predictor, X_tf.columns.tolist(), X_tf)
        
        return pipeline, predictor, timeline, explainer, df, X
    except Exception as e:
        pytest.fail(f"Failed to load artifacts: {e}")

# ==========================================
# MODULE 8: EXTREME MISSINGNESS & IMPUTATION STABILITY
# ==========================================
def test_empty_file_bureaucracy(artifacts):
    pipeline, predictor, _, _, _, base_X = artifacts
    
    # Construct an "empty" payload with 90% missing data
    empty_payload = base_X.iloc[[0]].copy()
    for col in empty_payload.columns:
        if pd.api.types.is_bool_dtype(empty_payload[col]):
            empty_payload[col] = empty_payload[col].astype(object)
            
    for col in empty_payload.columns:
        if pd.api.types.is_numeric_dtype(empty_payload[col]) and not pd.api.types.is_bool_dtype(empty_payload[col]):
            if np.random.rand() < 0.9:
                empty_payload.loc[0, col] = np.nan
        else:
            empty_payload.loc[0, col] = "Unknown"
            
    try:
        X_tf = pipeline.transform(empty_payload)
    except Exception as e:
        pytest.fail(f"Pipeline crashed on mostly missing data: {e}")
        
    assert not X_tf.isna().any().any(), "Imputation failed to clear all NaNs."
    
    # Inference
    preds = predictor.predict(X_tf)
    prob = preds['delay_probability'][0]
    
    assert prob > 0.0 and prob < 100.0, "Probability fell back to extreme 0 or 1 on missing data."
    assert 'risk_tier' in preds

def test_unseen_categorical_avalanche(artifacts):
    pipeline, predictor, _, _, _, base_X = artifacts
    
    payload = base_X.iloc[[0]].copy()
    payload['state'] = "New_State"
    payload['terrain_type'] = "Martian"
    payload['project_type'] = "Space_Elevator"
    
    try:
        X_tf = pipeline.transform(payload)
    except Exception as e:
        pytest.fail(f"Pipeline crashed on unseen categories: {e}")
        
    assert not X_tf.isna().any().any(), "Unseen categories caused NaN propagation."
    
    try:
        preds = predictor.predict(X_tf)
        assert preds is not None
    except Exception as e:
        pytest.fail(f"Predictor crashed on unseen category inference: {e}")

# ==========================================
# MODULE 9: EXPLAINABILITY STABILITY & FALLBACK (XAI)
# ==========================================
def test_shap_feature_perturbation_stability(artifacts):
    pipeline, _, _, explainer, _, base_X = artifacts

    # Run explanation on baseline payload
    P_base = base_X.iloc[[0]].copy()
    X_tf_base = pipeline.transform(P_base)
    base_explanation = explainer.explain(X_tf_base)

    assert "risk_drivers" in base_explanation
    base_top3 = [rd["feature"] for rd in base_explanation["risk_drivers"][:3]]
    assert len(base_top3) > 0

    # Add continuous feature jitter within +/- 0.1% to numerical inputs
    P_perturbed = P_base.copy()
    np.random.seed(42)
    for col in P_perturbed.select_dtypes(include=[np.number]).columns:
        jitter = np.random.uniform(-0.001, 0.001)
        P_perturbed[col] = P_perturbed[col] * (1.0 + jitter)

    X_tf_pert = pipeline.transform(P_perturbed)
    pert_explanation = explainer.explain(X_tf_pert)

    pert_top3 = [rd["feature"] for rd in pert_explanation["risk_drivers"][:3]]

    # Assert that top-3 identified risk drivers remain consistent (rank stability)
    overlap = len(set(base_top3).intersection(set(pert_top3)))
    assert overlap >= 2, f"Rank stability violated under 0.1% jitter: base={base_top3}, pert={pert_top3}"


def test_lime_fallback_schema_consistency(artifacts):
    pipeline, predictor, _, _, _, base_X = artifacts

    P_base = base_X.iloc[[0]].copy()
    X_tf = pipeline.transform(P_base)

    # Test graceful degradation path when SHAP calculation is disabled/fails
    fallback_explainer = DualParadigmExplainer(predictor, list(X_tf.columns), background_data=None)
    fallback_explainer.tree_models = {}  # Disable TreeSHAP to trigger fallback

    payload = fallback_explainer.explain(X_tf)

    # Assert returned payload adheres strictly to the primary explanation schema
    assert "global_importance_approx" in payload, "Missing global_importance_approx"
    assert "local_explanation_full" in payload, "Missing local_explanation_full"
    assert "category_breakdown" in payload, "Missing category_breakdown"
    assert "risk_drivers" in payload, "Missing risk_drivers"

    for rd in payload["risk_drivers"]:
        assert "feature" in rd
        assert "impact_score" in rd
        assert "direction" in rd
        assert "source" in rd
        assert np.isfinite(rd["impact_score"])
        assert 0.0 <= rd["impact_score"] <= 1.0

    for item in payload["local_explanation_full"]:
        assert "feature" in item
        assert "value" in item
        assert "shap_impact" in item
        assert "attention" in item
        assert "unified_score" in item
        assert np.isfinite(item["unified_score"])

    bd = payload["category_breakdown"]
    assert pytest.approx(sum(bd.values()), abs=1e-3) == 1.0

# ==========================================
# MODULE 10: ALGORITHMIC FAIRNESS & REGIONAL BIAS
# ==========================================
def test_regional_penalty_check(artifacts):
    pipeline, predictor, _, _, _, base_X = artifacts
    
    # Synthetic templates based on a real record
    template = base_X.iloc[[0]].copy()
    
    states = ["Maharashtra", "Odisha"]
    means = {}
    
    for state in states:
        synthetic_group = pd.concat([template]*100, ignore_index=True)
        synthetic_group['state'] = state
        
        X_tf = pipeline.transform(synthetic_group)
        preds = predictor.predict(X_tf)
        means[state] = np.mean(preds['delay_probability'])
        
    diff = abs(means["Maharashtra"] - means["Odisha"])
    assert diff <= 15.0, f"Potential regional bias detected! Diff = {diff:.2f}%"

def test_intersectional_group_fairness(artifacts):
    pipeline, predictor, _, _, _, base_X = artifacts
    template = base_X.iloc[[0]].copy()
    
    groups = [
        ("Maharashtra", "Urban"),
        ("Maharashtra", "Forest_Eco_Sensitive"),
        ("Bihar", "Urban"),
        ("Bihar", "Forest_Eco_Sensitive")
    ]
    
    means = []
    for state, terrain in groups:
        sg = pd.concat([template]*100, ignore_index=True)
        sg['state'] = state
        sg['terrain_type'] = terrain
        X_tf = pipeline.transform(sg)
        means.append(np.mean(predictor.predict(X_tf)['delay_probability']))
        
    cv = np.std(means) / np.mean(means)
    assert cv <= 1.05, f"Intersectional bias coefficient of variation too high: {cv:.3f}"

# ==========================================
# MODULE 11: SERIALIZATION & REPRODUCIBILITY
# ==========================================
def test_exact_float64_reproducibility(artifacts):
    pipeline, predictor, _, _, _, base_X = artifacts
    
    P_base = base_X.iloc[[0]].copy()
    X_tf_1 = pipeline.transform(P_base)
    prob_1 = predictor.predict(X_tf_1)['delay_probability'][0]
    
    # Simulate reload
    pipeline_reloaded = joblib.load('pipeline.joblib')
    predictor_reloaded = HybridRiskPredictor.load('ensemble.joblib')
    
    X_tf_2 = pipeline_reloaded.transform(P_base)
    prob_2 = predictor_reloaded.predict(X_tf_2)['delay_probability'][0]
    
    assert math.isclose(prob_1, prob_2, rel_tol=1e-8, abs_tol=1e-8), "Float64 reproducibility failed after reload."

def test_pipeline_state_consistency(artifacts):
    pipeline, predictor, timeline, _, _, _ = artifacts
    import gc
    gc.collect()

    p2 = joblib.load('pipeline.joblib')
    # Check tracker features
    assert list(pipeline.named_steps['tracker'].feature_names_in_) == list(p2.named_steps['tracker'].feature_names_in_)
    del p2
    gc.collect()

    pred2 = HybridRiskPredictor.load('ensemble.joblib')
    # Check ensemble features
    assert len(predictor.classifier.estimators) == len(pred2.classifier.estimators)
    del pred2
    gc.collect()

    # Check timeline models
    assert os.path.exists('timeline.joblib') and hasattr(timeline, 'rsf')

# ==========================================
# MODULE 12: HIGH-THROUGHPUT BATCH INFERENCE
# ==========================================
@pytest.mark.timeout(60)
def test_national_infrastructure_load(artifacts):
    pipeline, predictor, _, _, _, base_X = artifacts
    
    # Sample 10000 records
    n_samples = 10000
    indices = np.random.choice(base_X.index, size=n_samples, replace=True)
    df_load = base_X.loc[indices].reset_index(drop=True)
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    
    start_time = time.time()
    
    X_tf = pipeline.transform(df_load)
    preds = predictor.predict(X_tf)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    mem_after = process.memory_info().rss
    mem_increase = (mem_after - mem_before) / mem_before
    
    assert total_time <= 3.0, f"Batch latency too high: {total_time:.2f}s for 10k records"
    assert len(preds['delay_probability']) == n_samples
    assert mem_increase <= 0.15, f"Memory leak detected! Increase: {mem_increase*100:.1f}%"

# ==========================================
# MODULE 13: MODEL DRIFT & MONITORING READINESS
# ==========================================
def calculate_psi(expected, actual, bins=10):
    """Calculate Population Stability Index (PSI)"""
    expected_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=bins)[0] / len(actual)
    
    # Replace 0s to avoid div by zero
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi

def test_input_data_drift_detection(artifacts):
    pipeline, _, _, _, _, base_X = artifacts
    
    X_train_tf = pipeline.transform(base_X.head(1000))
    
    drift_X = base_X.head(1000).copy()
    drift_X['land_area_hectares'] = drift_X['land_area_hectares'] + (2 * drift_X['land_area_hectares'].std())
    drift_X['estimated_cost_inr_crore'] = drift_X['estimated_cost_inr_crore'] + (2 * drift_X['estimated_cost_inr_crore'].std())
    
    X_drift_tf = pipeline.transform(drift_X)
    
    psi_scores = []
    for col in ['land_area_hectares', 'estimated_cost_inr_crore']:
        if col in X_train_tf.columns:
            psi = calculate_psi(X_train_tf[col], X_drift_tf[col])
            psi_scores.append(psi)
            
    # As per prompt, just a warning or soft check
    high_drift = sum(p > 0.1 for p in psi_scores)
    # We assert that the metric is calculated, though high_drift might be triggered.
    # The requirement: "If not, the test should warn ... but may not fail"
    if high_drift > 0:
        print("WARNING: High data drift detected via PSI.")
    assert True

def test_prediction_drift_over_time(artifacts):
    pipeline, predictor, _, _, _, base_X = artifacts
    
    train_prob = np.mean(predictor.predict(pipeline.transform(base_X.head(1000)))['delay_probability'])
    
    # Time-shifted
    future_X = base_X.head(1000).copy()
    future_X['project_start_year'] = 2025
    
    future_prob = np.mean(predictor.predict(pipeline.transform(future_X))['delay_probability'])
    
    diff = abs(train_prob - future_prob)
    assert diff <= 5.0, f"Prediction drift exceeded 5%: {diff:.2f}%"

# ==========================================
# REPORT GENERATION HOOK
# ==========================================
test_results = {}

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        test_results[item.name] = {
            "status": "PASS" if rep.passed else "FAIL",
            "duration_secs": rep.duration
        }

def pytest_sessionfinish(session, exitstatus):
    """Write test_summary.json at the end"""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(test_results),
        "passed": sum(1 for r in test_results.values() if r["status"] == "PASS"),
        "failed": sum(1 for r in test_results.values() if r["status"] == "FAIL"),
        "results": test_results
    }
    
    with open("test_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
