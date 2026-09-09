import urllib.request
import json

def test_endpoint(url, method='GET', data=None, headers=None):
    headers = headers or {}
    req_body = None
    if data:
        req_body = json.dumps(data).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=req_body, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode('utf-8'))

# 1. Health endpoint
s, h = test_endpoint('http://localhost:8000/model/health')
print(f"GET /model/health: status={s}")
print(f"  Version: {h.get('version')}")
print(f"  Timestamp: {h.get('timestamp')}")
print(f"  NPU Provider: {h.get('npu_provider')}")
print(f"  Training Size: {h.get('training_size')}")
print(f"  Metrics: C-Index={h.get('metrics', {}).get('c_index')}, AUC={h.get('metrics', {}).get('auc')}, ECE={h.get('metrics', {}).get('ece')}")
print(f"  Gate Status: C-Index passed={h.get('metrics', {}).get('c_index_passed')}, AUC passed={h.get('metrics', {}).get('auc_passed')}, ECE passed={h.get('metrics', {}).get('ece_passed')}")

# 2. Drift endpoint
s, d = test_endpoint('http://localhost:8000/model/drift')
print(f"\nGET /model/drift: status={s}")
print(f"  Total Monitored: {d.get('total_features_monitored')}")
print(f"  Max PSI: {d.get('max_psi')}")
print(f"  Mean PSI: {d.get('mean_psi')}")
print(f"  Auto-retrain Triggered: {d.get('auto_retrain_triggered')}")

# 3. Predict sample with SHAP waterfall, RSF survival curve, Prescriptive Engine
payload = {
    'project_id': 'TEST-E2E-PROJ',
    'project_type': 'Highway',
    'state': 'Maharashtra',
    'district': 'Pune',
    'terrain_type': 'Plain',
    'land_area_hectares': 150.0,
    'estimated_cost_inr_crore': 1200.0,
    'affected_families_count': 350,
    'title_dispute_rate_percent': 14.5,
    'local_protest_flag': False,
    'compensation_multiplier_demand': 1.8,
    'sia_approval_status': 'Approved',
    'forest_clearance_status': 'Stage 1 Approved',
    'fund_disbursement_percent': 65.0
}
s, p = test_endpoint('http://localhost:8000/predict', method='POST', data=payload, headers={'X-API-Key': 'super-secret-token'})
print(f"\nPOST /predict: status={s}")
print(f"  Delay Probability: {p.get('predictions', {}).get('delay_probability'):.4f}")
print(f"  Risk Tier: {p.get('predictions', {}).get('calibrated_risk_tier')}")
print(f"  Predicted Delay Days: {p.get('predictions', {}).get('predicted_delay_days')}")
print(f"  Uno C-Index: {p.get('predictions', {}).get('uno_c_index')}")
print(f"  SHAP Risk Drivers: {len(p.get('explainability', {}).get('top_risk_drivers', []))} drivers returned")
surv_curve = p.get('survival_curve', [])
surv_count = len(surv_curve) if isinstance(surv_curve, list) else len(surv_curve.get('timeline_days', []))
print(f"  RSF Survival Curve: {surv_count} horizons")
print(f"  Prescriptive Mitigations: {len(p.get('prescriptive_actions', []))} actions returned")

print("\n[SUCCESS] API verification complete! All endpoints operational.")
