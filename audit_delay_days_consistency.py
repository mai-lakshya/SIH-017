import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import joblib

from compat import apply_all_patches
apply_all_patches()

from hybrid_model import HybridRiskPredictor
from timeline_predictor import NonLinearTimelinePredictor
from recommendation_engine import RecommendationEngine, calculate_roi_for_recommendation
from risk_analysis_system import RiskAnalysisSystem

def run_consistency_audit():
    print("=" * 80)
    print("PART D: DELAY-DAYS CONSISTENCY & RECOMMENDATION ENGINE AUDIT")
    print("=" * 80)

    pipeline = joblib.load('pipeline.joblib')
    hybrid = HybridRiskPredictor.load('ensemble.joblib')
    timeline = joblib.load('timeline.joblib')
    df = pd.read_csv('indian_infrastructure_projects_dataset.csv')

    X = df.drop(columns=['delay_binary_label', 'Actual_Delay_Days', 'CRS', 'project_index'], errors='ignore')
    X_tf = pipeline.transform(X.head(100))

    hybrid_preds = hybrid.predict(X_tf)
    hybrid_days = hybrid_preds['predicted_delay_days']

    timeline_days = timeline.predict_time_to_delay(X_tf)
    timeline_median = timeline.get_dynamic_risk_threshold(X_tf)

    abs_diff = np.abs(hybrid_days - timeline_days)
    rel_diff = abs_diff / np.maximum(timeline_days, 1.0)

    rho_spearman, p_spearman = spearmanr(hybrid_days, timeline_days)
    r_pearson, p_pearson = pearsonr(hybrid_days, timeline_days)

    print(f"\n--- 1. DELAY-DAYS CROSS-MODEL COMPARISON (N=100) ---")
    print(f"Hybrid Regressor Days: Mean = {np.mean(hybrid_days):.2f}, Median = {np.median(hybrid_days):.2f}, Std = {np.std(hybrid_days):.2f}")
    print(f"Timeline Survival Days: Mean = {np.mean(timeline_days):.2f}, Median = {np.median(timeline_days):.2f}, Std = {np.std(timeline_days):.2f}")
    print(f"Mean Absolute Disagreement: {np.mean(abs_diff):.2f} days")
    print(f"Median Absolute Disagreement: {np.median(abs_diff):.2f} days")
    print(f"Max Absolute Disagreement: {np.max(abs_diff):.2f} days")
    print(f"Pearson Correlation r: {r_pearson:.4f} (p = {p_pearson:.4e})")
    print(f"Spearman Rank Correlation rho: {rho_spearman:.4f} (p = {p_spearman:.4e})")

    # Trace authoritative split in risk_analysis_system.py
    sys = RiskAnalysisSystem(pipeline_path='pipeline.joblib', ensemble_path='ensemble.joblib', timeline_path='timeline.joblib')
    sample_raw = df.iloc[[0]].drop(columns=['delay_binary_label', 'Actual_Delay_Days', 'CRS', 'project_index'], errors='ignore')
    sys_out = sys.predict(sample_raw)
    
    headline_hybrid_days = sys_out['predictions']['predicted_delay_days']
    headline_timeline_median = sys_out['timeline']['median_survival_days']
    print(f"\n--- 2. ORCHESTRATION CONTRACT IN RISK_ANALYSIS_SYSTEM ---")
    print(f"Predictions block 'predicted_delay_days': {headline_hybrid_days} (from Hybrid Stacking Regressor)")
    print(f"Timeline block 'median_survival_days': {headline_timeline_median} (from RSF NonLinearTimelinePredictor)")

    # 3. Re-verify C4 (Direction-aware mitigation simulation) end-to-end
    print(f"\n--- 3. END-TO-END VERIFICATION OF C4 DIRECTION-AWARE MITIGATION ---")
    high_risk_idx = None
    for i in range(len(df)):
        if df.iloc[i].get('title_dispute_rate_percent', 0) > 20 and df.iloc[i].get('local_protest_flag', False):
            high_risk_idx = i
            break
    if high_risk_idx is None:
        high_risk_idx = 0

    high_risk_row = df.iloc[[high_risk_idx]].copy()
    high_risk_tf = pipeline.transform(high_risk_row.drop(columns=['delay_binary_label', 'Actual_Delay_Days', 'CRS', 'project_index'], errors='ignore'))
    
    orig_hybrid_days = float(hybrid.predict(high_risk_tf)['predicted_delay_days'][0])
    orig_prob = float(hybrid.predict(high_risk_tf)['delay_probability'][0])
    
    project_cost_cr = float(high_risk_row['estimated_cost_inr_crore'].values[0])
    project_cost_inr = project_cost_cr * 10_000_000
    delay_cost_per_day = max(100_000, (project_cost_inr * 0.12) / 365)

    rec_engine = RecommendationEngine()
    drivers = [
        ('title_dispute_rate_percent', 0.85),
        ('local_protest_flag', 0.70),
        ('fund_disbursement_percent', 0.60)
    ]
    meta = {
        'state': str(high_risk_row['state'].values[0]),
        'terrain_type': str(high_risk_row['terrain_type'].values[0]),
        'title_dispute_rate_percent': float(high_risk_row['title_dispute_rate_percent'].values[0]),
        'estimated_cost_inr_crore': project_cost_cr
    }

    recs = rec_engine.generate_recommendations(drivers, meta)
    print(f"Selected High-Risk Project #{high_risk_idx} (Cost: {project_cost_cr} Cr, Base Delay: {orig_hybrid_days:.1f} days, Base Prob: {orig_prob*100:.1f}%)")
    print(f"Generated {len(recs)} Recommendations:")

    c4_verifications = []
    for idx, rec in enumerate(recs, 1):
        roi = calculate_roi_for_recommendation(
            rec,
            project_cost=project_cost_inr,
            delay_cost_per_day=delay_cost_per_day,
            model=hybrid,
            X_sample=high_risk_tf
        )
        days_saved = roi['estimated_delay_days_saved']
        cost_savings = roi['cost_savings']
        roi_pct = roi['roi_percentage']
        impl_cost = roi.get('implementation_cost', delay_cost_per_day)

        # Expected cost savings = days_saved * delay_cost_per_day (within 0.05 day display rounding tolerance)
        expected_savings = days_saved * delay_cost_per_day
        savings_consistent = abs(cost_savings - expected_savings) <= (0.05 * delay_cost_per_day + 1.0)

        # Expected ROI = ((cost_savings - impl_cost) / impl_cost) * 100
        expected_roi = ((cost_savings - impl_cost) / impl_cost) * 100.0
        roi_consistent = abs(roi_pct - expected_roi) < 1.0

        print(f"  Rec {idx}: {rec['issue']}")
        print(f"    Days Saved: {days_saved:.1f} days")
        print(f"    Cost Savings: {cost_savings / 1e7:.2f} Cr (Expected: {expected_savings / 1e7:.2f} Cr, Consistent: {savings_consistent})")
        print(f"    Implementation Cost: {impl_cost / 1e7:.4f} Cr")
        print(f"    ROI: {roi_pct:.1f}% (Expected: {expected_roi:.1f}%, Consistent: {roi_consistent})")

        c4_verifications.append({
            'recommendation_id': rec.get('recommendation_id'),
            'issue': rec.get('issue'),
            'days_saved': days_saved,
            'cost_savings_inr': cost_savings,
            'cost_savings_cr': cost_savings / 1e7,
            'impl_cost_inr': impl_cost,
            'roi_percentage': roi_pct,
            'savings_consistent': savings_consistent,
            'roi_consistent': roi_consistent
        })

    all_consistent = all(v['savings_consistent'] and v['roi_consistent'] for v in c4_verifications)
    print(f"\nC4 End-to-End Consistency Check: {'PASS' if all_consistent else 'FAIL'}")

    results = {
        'cross_model_comparison_n': 100,
        'hybrid_mean_days': float(np.mean(hybrid_days)),
        'hybrid_median_days': float(np.median(hybrid_days)),
        'timeline_mean_days': float(np.mean(timeline_days)),
        'timeline_median_days': float(np.median(timeline_days)),
        'mean_absolute_disagreement_days': float(np.mean(abs_diff)),
        'median_absolute_disagreement_days': float(np.median(abs_diff)),
        'max_absolute_disagreement_days': float(np.max(abs_diff)),
        'pearson_r': float(r_pearson),
        'pearson_pvalue': float(p_pearson),
        'spearman_rho': float(rho_spearman),
        'spearman_pvalue': float(p_spearman),
        'c4_verified': all_consistent,
        'c4_verifications': c4_verifications
    }

    with open('delay_days_consistency_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    run_consistency_audit()
