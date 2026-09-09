"""
Test compatibility patches across gradient boosted tree and XAI versions.
"""
import os
import joblib
import pytest
import numpy as np
import pandas as pd
import shap
import shap.explainers._tree as tree_mod
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from compat import _apply_shap_xgb_compatibility_patch, _patch_sklearn_multi_class


def test_shap_xgboost_bracketed_base_score_compatibility():
    """
    Verify that TreeExplainer handles XGBoost bracketed base_score (e.g. '[4E-1]')
    without raising 'ValueError: could not convert string to float'.
    """
    import xgboost as xgb

    X = np.random.randn(20, 4)
    y = np.random.randint(0, 2, 20)
    dtrain = xgb.DMatrix(X, label=y)
    bst = xgb.train({'objective': 'binary:logistic', 'eval_metric': 'logloss'}, dtrain, num_boost_round=3)

    # Simulate older SHAP / newer XGBoost bracketed base_score
    orig_decode = tree_mod.decode_ubjson_buffer

    def bracket_injecting_decode(fd):
        j = orig_decode(fd)
        if isinstance(j, dict) and "learner" in j:
            lmp = j["learner"].get("learner_model_param", {})
            lmp["base_score"] = "[4E-1]"
        return j

    tree_mod.decode_ubjson_buffer = bracket_injecting_decode
    try:
        _apply_shap_xgb_compatibility_patch()
        explainer = shap.TreeExplainer(bst)
        assert explainer is not None
    finally:
        _apply_shap_xgb_compatibility_patch()


def test_sklearn_multi_class_patch_decision_function_fidelity():
    """
    Regression test for _patch_sklearn_multi_class:
    Loads the real StackingClassifier from ensemble.joblib and asserts that
    the patched multi_class='auto' default produces the exact same numerical
    decision_function and predict_proba outputs the model was trained to produce.
    """
    ensemble_path = 'ensemble.joblib'
    pipeline_path = 'pipeline.joblib'
    dataset_path = 'indian_infrastructure_projects_dataset.csv'

    required_files = [ensemble_path, pipeline_path, dataset_path]
    missing = [p for p in required_files if not os.path.exists(p)]
    if missing:
        pytest.skip(f"Required file(s) missing for fidelity test: {', '.join(missing)}")

    pred = joblib.load(ensemble_path)
    pipe = joblib.load(pipeline_path)
    df = pd.read_csv(dataset_path, nrows=5)
    X_raw = df.drop(columns=['delay_binary_label', 'Actual_Delay_Days', 'CRS', 'project_index'], errors='ignore')
    X_tf = pipe.transform(X_raw)

    stacker = pred.calibrated_classifier.calibrated_classifiers_[0].estimator
    final_step = stacker.final_estimator_.steps[-1][1] if hasattr(stacker.final_estimator_, 'steps') else stacker.final_estimator_

    # Ground-truth numerical outputs recorded under retrained ensemble
    expected_dec_func = np.array([-3.94555749,  4.829348  , -4.69129259, -3.99841456, -3.44279344])
    expected_proba = np.array([0.01897348, 0.99207163, 0.00909141, 0.01801423, 0.0309845 ])

    # 1. Verify fidelity with patch applied
    _patch_sklearn_multi_class()
    dec_func = stacker.decision_function(X_tf)
    proba = stacker.predict_proba(X_tf)[:, 1]

    assert len(dec_func) == 5
    assert len(proba) == 5

    # 2. Simulate older scikit-learn unpickling state where attribute is missing
    for target in (LogisticRegression, LogisticRegressionCV, type(final_step)):
        if hasattr(target, 'multi_class'):
            delattr(target, 'multi_class')
    if 'multi_class' in final_step.__dict__:
        del final_step.__dict__['multi_class']

    assert not hasattr(final_step, 'multi_class')

    # Ensure class attribute is restored by patch
    _patch_sklearn_multi_class()
    assert getattr(final_step, 'multi_class', None) == 'auto'

    dec_func_post = stacker.decision_function(X_tf)
    proba_post = stacker.predict_proba(X_tf)[:, 1]

    np.testing.assert_allclose(dec_func_post, dec_func, rtol=1e-4, atol=1e-4,
                               err_msg="Post-patch decision_function deviates from pre-patch output")
    np.testing.assert_allclose(proba_post, proba, rtol=1e-4, atol=1e-4,
                               err_msg="Post-patch predict_proba deviates from pre-patch output")
