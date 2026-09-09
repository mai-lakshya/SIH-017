"""
End-to-End Verification Test for NEXUS-XAI Continuous Learning Pipeline
=====================================================================
1. Ingest 50 synthetic project records.
2. Compute feature-by-feature PSI drift.
3. Retrain stacking ensemble + RSF + Platt calibration + TreeSHAP with n_jobs=-1.
4. Verify Validation Gate (C-Index >= 0.88, ECE <= 0.10, AUC >= 0.85).
5. Verify versioning (keep last 3 versions) + model_card.json.
6. Verify NPU execution provider auto-detection & ONNX inference.
7. Verify existing SHAP waterfall, RSF curve, and prescriptive engine integrity.
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd
import joblib

from continuous_learning import (
    ingest_project_records,
    DriftDetector,
    RetrainingOrchestrator,
    ValidationGate,
    ModelVersionManager,
    NPUEngine,
    detect_npu_provider,
    DATA_STORE_PATH,
    MODELS_DIR
)
from risk_analysis_system import RiskAnalysisSystem

STATES = ["Maharashtra", "Telangana", "Karnataka", "Uttar Pradesh", "Gujarat", "Tamil Nadu", "West Bengal", "Bihar"]
PROJECT_TYPES = ["Highway", "Railway", "Renewable_Energy"]
TERRAINS = ["Plain", "Hilly", "Coastal", "Urban", "Desert"]
SIA_STATUSES = ["Approved", "Pending", "Exempted"]
FOREST_STATUSES = ["Approved", "Stage 1 Approved", "Pending", "Not_Required"]

STATE_DISTRICTS = {
    "Bihar": ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Begusarai"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar"],
    "Karnataka": ["Bengaluru Urban", "Mysuru", "Belagavi", "Kalaburagi", "Dharwad"],
    "Maharashtra": ["Pune", "Nagpur", "Nashik", "Thane", "Chhatrapati Sambhajinagar"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirappalli"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam"],
    "Uttar Pradesh": ["Lucknow", "Kanpur Nagar", "Varanasi", "Agra", "Prayagraj"],
    "West Bengal": ["Kolkata", "Howrah", "North 24 Parganas", "Paschim Bardhaman", "Darjeeling"],
}


def generate_50_synthetic_records() -> list:
    """Generates 50 realistic infrastructure project records."""
    random.seed(42)
    np.random.seed(42)
    records = []
    for i in range(50):
        cost = round(random.uniform(250.0, 5000.0), 2)
        area = round(random.uniform(30.0, 450.0), 2)
        families = int(random.uniform(50, 1200))
        dispute = round(random.uniform(0.0, 25.0), 2)
        disburse = round(random.uniform(5.0, 95.0), 1)
        protest = random.choice([True, False, False, False])
        state = random.choice(STATES)
        p_type = random.choice(PROJECT_TYPES)
        terrain = random.choice(TERRAINS)
        sia = random.choice(SIA_STATUSES)
        forest = random.choice(FOREST_STATUSES)

        # Synthetic ground truth label consistent with risk factors
        is_delayed = 1 if (dispute > 15.0 or protest or sia == "Pending" or forest == "Pending" or disburse < 20.0) else 0
        delay_days = round(random.uniform(90.0, 420.0) if is_delayed else random.uniform(30.0, 90.0), 1)
        crs = round(random.uniform(50.0, 90.0) if is_delayed else random.uniform(10.0, 45.0), 1)

        dists = STATE_DISTRICTS.get(state, ["Patna"])
        rec = {
            "project_id": f"TEST-SYNTH-{1000 + i}",
            "state": state,
            "district": dists[i % len(dists)],
            "project_type": p_type,
            "terrain_type": terrain,
            "land_area_hectares": area,
            "estimated_cost_inr_crore": cost,
            "affected_families_count": families,
            "title_dispute_rate_percent": dispute,
            "local_protest_flag": protest,
            "compensation_multiplier_demand": round(random.uniform(1.2, 2.5), 2),
            "sia_approval_status": sia,
            "section_11_notification_days": int(delay_days),
            "forest_clearance_status": forest,
            "fund_disbursement_percent": disburse,
            "delay_binary_label": is_delayed,
            "CRS": crs,
            "project_start_year": random.choice([2021, 2022, 2023, 2024])
        }
        records.append(rec)
    return records


def run_pipeline_verification():
    print("\n" + "=" * 76)
    print("      NEXUS-XAI CONTINUOUS LEARNING END-TO-END VERIFICATION SUITE")
    print("=" * 76)

    # -------------------------------------------------------------
    # 1. HARDWARE & NPU AUTO-DETECTION TEST
    # -------------------------------------------------------------
    print("\n[STEP 1/7] Testing NPU Provider Auto-Detection...")
    selected_prov, all_provs = detect_npu_provider()
    print(f"  -> All Detected Providers: {all_provs}")
    print(f"  -> Active Hardware Accelerated Provider: {selected_prov}")
    assert selected_prov in all_provs, "Selected provider must be in available providers list"

    # -------------------------------------------------------------
    # 2. INGESTION TEST (50 Synthetic Records)
    # -------------------------------------------------------------
    print("\n[STEP 2/7] Testing Ingestion with 50 Synthetic Records...")
    synth_records = generate_50_synthetic_records()
    assert len(synth_records) == 50, "Expected exactly 50 synthetic records"

    ingest_result = ingest_project_records(synth_records, data_store_path=DATA_STORE_PATH)
    print(f"  -> Ingested: {ingest_result['ingested_count']} records")
    print(f"  -> Total Data Store Size: {ingest_result['total_store_size']} projects")
    assert ingest_result["status"] == "success"
    assert ingest_result["ingested_count"] == 50

    # -------------------------------------------------------------
    # 3. FEATURE-BY-FEATURE PSI DRIFT DETECTION TEST
    # -------------------------------------------------------------
    print("\n[STEP 3/7] Testing Population Stability Index (PSI) Drift Detection...")
    df_all = pd.read_csv(DATA_STORE_PATH)
    recent_50 = pd.DataFrame(synth_records)
    detector = DriftDetector(baseline_df=df_all.iloc[:-50])
    drift_report = detector.evaluate_drift(incoming_df=recent_50)

    print(f"  -> Total Features Monitored: {drift_report['total_features_monitored']}")
    print(f"  -> Max PSI: {drift_report['max_psi']:.4f}")
    print(f"  -> Mean PSI: {drift_report['mean_psi']:.4f}")
    print(f"  -> Retrain Triggered (PSI > 0.20): {drift_report['auto_retrain_triggered']}")

    # Print top 5 features by PSI
    sorted_feats = sorted(drift_report['feature_reports'].items(), key=lambda x: x[1]['psi'], reverse=True)
    print("  -> Top 5 Features by PSI:")
    for feat_name, info in sorted_feats[:5]:
        print(f"     * {feat_name:<32} PSI: {info['psi']:.4f} [{info['badge']}]")

    assert drift_report['total_features_monitored'] > 0
    assert "feature_reports" in drift_report

    # -------------------------------------------------------------
    # 4. PARALLELIZED RETRAINING & VALIDATION GATE
    # -------------------------------------------------------------
    print("\n[STEP 4/7] Testing Parallel Retraining (n_jobs=-1) & Validation Gate...")
    orchestrator = RetrainingOrchestrator(data_store_path=DATA_STORE_PATH)
    retrain_res = orchestrator.run_retrain_cycle(trigger_reason="e2e_verification_test")

    print(f"  -> Candidate Model Version: {retrain_res['version']}")
    print(f"  -> Gate Decision Promoted: {retrain_res['promoted']}")
    print(f"  -> Candidate Metrics: {retrain_res['metrics']}")

    # Validate gate criteria
    m = retrain_res["metrics"]
    assert m["auc"] >= 0.85, f"AUC ({m['auc']}) must be >= 0.85"
    assert m["ece"] <= 0.10, f"ECE ({m['ece']}) must be <= 0.10"
    assert m["c_index"] >= 0.88, f"C-Index ({m['c_index']}) must be >= 0.88"
    assert retrain_res["promoted"] is True, "Candidate satisfying all criteria must be promoted"

    # -------------------------------------------------------------
    # 5. MODEL VERSIONING & 3-VERSION PRUNING TEST
    # -------------------------------------------------------------
    print("\n[STEP 5/7] Testing Model Versioning & 3-Version Retention...")
    vm = ModelVersionManager(root_models_dir=MODELS_DIR)
    history = vm.get_version_history()
    print(f"  -> Retained Model Versions: {len(history)}")
    for v in history:
        print(f"     * Version {v.get('version')} ({v.get('timestamp')}) - Size: {v.get('training_size')}")
    assert len(history) <= 3, f"Must retain at most 3 versions, found {len(history)}"

    # -------------------------------------------------------------
    # 6. NPU ACCELERATION & ONNX BATCH INFERENCE TEST
    # -------------------------------------------------------------
    print("\n[STEP 6/7] Testing ONNX Export & Hardware Accelerated NPU Inference...")
    pipeline = joblib.load("pipeline.joblib")
    X_test_proc = pipeline.transform(recent_50.drop(columns=['delay_binary_label', 'section_11_notification_days', 'CRS', 'project_id'], errors='ignore'))
    
    onnx_path = "model.onnx"
    assert os.path.exists(onnx_path), f"ONNX model {onnx_path} must exist"
    
    npu_engine = NPUEngine(model_onnx_path=onnx_path)
    t0 = time.perf_counter()
    npu_preds = npu_engine.predict_batch_npu(X_test_proc.values)
    t_npu = (time.perf_counter() - t0) * 1000.0

    print(f"  -> NPU Inference Latency for 50 records: {t_npu:.2f} ms ({t_npu/50:.2f} ms/record)")
    print(f"  -> NPU Predictions Output Shape: {npu_preds.shape}")
    assert len(npu_preds) == 50, "Expected 50 batch predictions from NPU engine"

    # -------------------------------------------------------------
    # 7. REGRESSION TEST: SHAP WATERFALL, RSF CURVE, PRESCRIPTIONS
    # -------------------------------------------------------------
    print("\n[STEP 7/7] Verifying SHAP Waterfall, RSF Survival Curve & Prescriptive Engine...")
    system = RiskAnalysisSystem(
        pipeline_path='pipeline.joblib',
        ensemble_path='ensemble.joblib',
        timeline_path='timeline.joblib'
    )
    test_sample = recent_50.iloc[[0]].drop(columns=['delay_binary_label', 'section_11_notification_days', 'CRS', 'project_id'], errors='ignore')
    pred_result = system.predict(test_sample)

    # Verify SHAP waterfall data is intact
    assert "explanation" in pred_result or "explainability" in pred_result, "Missing explainability in prediction result"
    exp = pred_result.get("explanation", pred_result.get("explainability", {}))
    risk_drivers = exp.get("risk_drivers", exp.get("top_risk_drivers", []))
    assert len(risk_drivers) > 0, "Expected at least 1 SHAP risk driver"
    print(f"  -> SHAP Attributions Verified: {len(risk_drivers)} drivers returned")

    # Verify RSF survival curve is intact
    X_proc = system.pipeline.transform(test_sample)
    surv_funcs = system.timeline_predictor.rsf.predict_survival_function(X_proc)
    sample_horizons = np.array([90, 180, 270, 365, 540])
    surv_probs = surv_funcs[0](sample_horizons)
    assert len(surv_probs) == len(sample_horizons), "Invalid survival curve evaluation"
    assert all(0.0 <= p <= 1.0 for p in surv_probs), "Survival probabilities out of range"
    print(f"  -> RSF Survival Curve Verified: {len(sample_horizons)} time horizons evaluated successfully")

    # Verify Prescriptive Engine recommendations are intact
    recs = pred_result.get("recommendations", pred_result.get("prescriptive_actions", []))
    assert len(recs) > 0, "Expected at least 1 prescriptive action"
    print(f"  -> Prescriptive Engine Verified: {len(recs)} actionable mitigations returned")

    # Clean up test records from CSV data store so the dataset remains clean
    try:
        csv_file = "indian_infrastructure_projects_dataset.csv"
        if os.path.exists(csv_file):
            df_curr = pd.read_csv(csv_file, low_memory=False)
            df_pruned = df_curr[~df_curr['project_id'].str.contains('TEST-SYNTH', case=False, na=False)]
            if len(df_pruned) != len(df_curr):
                df_pruned.to_csv(csv_file, index=False)
                print(f"  -> Cleanup: Removed {len(df_curr) - len(df_pruned)} test records from dataset.")
    except Exception as e:
        print(f"  -> Cleanup note: {e}")

    print("\n" + "=" * 76)
    print("   [SUCCESS] ALL 7 CONTINUOUS LEARNING & NPU PIPELINE TESTS PASSED!")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    run_pipeline_verification()
