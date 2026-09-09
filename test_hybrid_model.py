import numpy as np
import pandas as pd
import pytest
from hybrid_model import HybridRiskPredictor, safe_logit

def test_safe_logit():
    # Should not be inf or -inf
    assert np.isfinite(safe_logit(0.0))
    assert np.isfinite(safe_logit(1.0))
    assert np.isfinite(safe_logit(0.5))

@pytest.fixture
def sample_data():
    np.random.seed(42)
    # 50 samples, 5 features
    X = pd.DataFrame(np.random.rand(50, 5), columns=[f'f{i}' for i in range(5)])
    y_cls = np.random.randint(0, 2, 50)
    y_crs = np.random.rand(50) * 100
    y_days = np.random.rand(50) * 1000
    return X, y_cls, y_crs, y_days

def test_hybrid_risk_predictor_fit_predict(sample_data):
    X, y_cls, y_crs, y_days = sample_data
    
    predictor = HybridRiskPredictor(random_state=42)
    predictor.fit(X, y_cls, y_crs, y_days)
    
    preds = predictor.predict(X, blend_monotonicity=False)
    
    assert 'delay_probability' in preds
    assert 'crs' in preds
    assert 'raw_crs' in preds
    assert 'predicted_crs' in preds
    assert 'adjusted_risk_index' in preds
    assert 'delay_days' in preds
    assert 'predicted_delay_days' in preds
    assert 'days_p10' in preds
    assert 'days_p90' in preds
    assert 'crs_p10' in preds
    assert 'crs_p90' in preds
    assert 'confidence_interval_90' in preds
    
    # Check probability bounds
    assert np.all((preds['delay_probability'] >= 0.0) & (preds['delay_probability'] <= 1.0))
    # Check CRS bounds
    assert np.all((preds['crs'] >= 0.0) & (preds['crs'] <= 100.0))
    # Check statutory days bounds
    assert np.all((preds['delay_days'] >= 30.0) & (preds['delay_days'] <= 730.0))
    
def test_hybrid_risk_predictor_monotonicity_blend(sample_data):
    X, y_cls, y_crs, y_days = sample_data
    
    predictor = HybridRiskPredictor(random_state=42)
    predictor.fit(X, y_cls, y_crs, y_days)
    
    preds = predictor.predict(X)
    
    # Check that calibrated crs is preserved and raw_crs is uncorrupted
    assert 'adjusted_risk_index' in preds
    assert 'crs' in preds
    assert 'predicted_crs' in preds
    np.testing.assert_allclose(preds['crs'], preds['raw_crs'])
    
    # Check that adjusted_risk_index applies the heuristic scaling
    expected_adjusted_risk = np.clip(preds['raw_crs'] * (0.5 + preds['delay_probability']), 0.0, 100.0)
    np.testing.assert_allclose(preds['adjusted_risk_index'], expected_adjusted_risk, rtol=1e-5)
    
    # Check that adjusted_delay_days applies heuristic scaling
    expected_adjusted_days = np.clip(preds['raw_delay_days'] * (0.5 + preds['delay_probability']), 30.0, 730.0)
    np.testing.assert_allclose(preds['adjusted_delay_days'], expected_adjusted_days, rtol=1e-5)
    
    # Check conformal bounds (P10 <= prediction <= P90)
    assert np.all(preds['days_p10'] <= preds['delay_days'] + 1e-5)
    assert np.all(preds['delay_days'] <= preds['days_p90'] + 1e-5)
    assert np.all(preds['crs_p10'] <= preds['crs'] + 1e-5)
    assert np.all(preds['crs'] <= preds['crs_p90'] + 1e-5)

def test_feature_leakage_guard():
    from survival_features import BASE_PIPELINE_FEATURES, SELECTED_SURVIVAL_FEATURES
    assert 'project_id' not in BASE_PIPELINE_FEATURES, "project_id must not be in BASE_PIPELINE_FEATURES"
    assert 'project_id' not in SELECTED_SURVIVAL_FEATURES, "project_id must not be in SELECTED_SURVIVAL_FEATURES"
    assert len(BASE_PIPELINE_FEATURES) == 27, f"Expected 27 features, got {len(BASE_PIPELINE_FEATURES)}"
    
    import joblib
    pipeline = joblib.load('pipeline.joblib')
    feature_names = getattr(pipeline, 'feature_names_in_', [])
    assert 'project_id' not in feature_names, "pipeline.joblib feature_names_in_ must not contain project_id"


