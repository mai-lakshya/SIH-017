import json
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import joblib

from compat import apply_all_patches
apply_all_patches()

from timeline_explainer import TimelinePermutationExplainer

def run_timeline_audit():
    print("=" * 80)
    print("TIMELINE PERMUTATION EXPLAINER (LOCAL MODE) FAITHFULNESS AUDIT")
    print("=" * 80)

    pipeline = joblib.load('pipeline.joblib')
    timeline = joblib.load('timeline.joblib')
    df = pd.read_csv('indian_infrastructure_projects_dataset.csv')

    X = df.drop(columns=['delay_binary_label', 'Actual_Delay_Days', 'CRS', 'project_index'], errors='ignore')
    X_tf = pipeline.transform(X.head(500))
    feature_names = X_tf.columns.tolist()

    X_bg = X_tf.iloc[:200].copy()
    neutral_medians = X_bg.median(numeric_only=True).to_dict()

    tl_explainer = TimelinePermutationExplainer(
        timeline_predictor=timeline,
        feature_names=feature_names,
        background_data=X_bg
    )

    n_eval = 50
    eval_df = X_tf.iloc[:n_eval].copy()

    top1_claimed, top1_measured, top1_dirs = [], [], []
    top3_claimed, top3_measured, top3_dirs = [], [], []
    telemetry = []

    for i in range(n_eval):
        row_orig = eval_df.iloc[[i]].copy()
        base_days = float(timeline.predict_time_to_delay(row_orig)[0])
        base_risk = float(timeline.predict(row_orig)[0])

        exp = tl_explainer.explain(row_orig, top_k=3, mode='local')
        drivers = exp['top_drivers']

        row_rec = {
            'row_index': i,
            'base_days': base_days,
            'base_risk': base_risk,
            'drivers': []
        }

        for rank, d in enumerate(drivers, start=1):
            feat = d['feature']
            claimed_impact = d['importance']
            claimed_dir = d['direction']

            orig_val = float(row_orig[feat].values[0])
            neut_val = float(neutral_medians.get(feat, 0.0))

            row_del = row_orig.copy()
            row_del[feat] = neut_val

            del_days = float(timeline.predict_time_to_delay(row_del)[0])
            del_risk = float(timeline.predict(row_del)[0])

            delta_days = base_days - del_days
            abs_delta_days = abs(delta_days)

            delta_risk = base_risk - del_risk
            abs_delta_risk = abs(delta_risk)

            if claimed_dir == 'increases_delay':
                dir_match = (delta_days > 0)
                risk_dir_match = (delta_risk > 0)
            else:
                dir_match = (delta_days < 0)
                risk_dir_match = (delta_risk < 0)

            is_true_pert = abs(orig_val - neut_val) > 1e-5

            d_info = {
                'rank': rank,
                'feature': feat,
                'claimed_impact': claimed_impact,
                'claimed_dir': claimed_dir,
                'orig_val': orig_val,
                'neut_val': neut_val,
                'base_days': base_days,
                'del_days': del_days,
                'delta_days': delta_days,
                'abs_delta_days': abs_delta_days,
                'dir_match': dir_match,
                'is_true_pert': is_true_pert
            }
            row_rec['drivers'].append(d_info)
            telemetry.append(d_info)

            top3_claimed.append(claimed_impact)
            top3_measured.append(abs_delta_days)
            top3_dirs.append(dir_match)

            if rank == 1:
                top1_claimed.append(claimed_impact)
                top1_measured.append(abs_delta_days)
                top1_dirs.append(dir_match)

    df_tel = pd.DataFrame(telemetry)
    n_pert = int(df_tel['is_true_pert'].sum())
    total_trials = len(df_tel)

    rho_t1, p_t1 = spearmanr(top1_claimed, top1_measured)
    rho_t3, p_t3 = spearmanr(top3_claimed, top3_measured)

    print(f"Total Trials: {total_trials}")
    print(f"True Perturbations: {n_pert} / {total_trials} ({n_pert / total_trials * 100:.1f}%)")
    print(f"Top-1 Driver Spearman rho (All N=50): {rho_t1:.4f} (p = {p_t1:.4e})")
    print(f"Top-3 Drivers Spearman rho (All N=150): {rho_t3:.4f} (p = {p_t3:.4e})")
    print(f"Top-1 Directional Fidelity (All N=50): {np.mean(top1_dirs) * 100:.1f}%")
    print(f"Top-3 Directional Fidelity (All N=150): {np.mean(top3_dirs) * 100:.1f}%")

    df_pert = df_tel[df_tel['is_true_pert']]
    rho_pert, p_pert = spearmanr(df_pert['claimed_impact'], df_pert['abs_delta_days'])
    dir_fid_pert = float(df_pert['dir_match'].mean() * 100)
    print(f"Top-3 Spearman rho on True Perturbations (N={len(df_pert)}): {rho_pert:.4f} (p = {p_pert:.4e})")
    print(f"Top-3 Directional Fidelity on True Perturbations (N={len(df_pert)}): {dir_fid_pert:.1f}%")

    df_t1_pert = df_tel[(df_tel['rank'] == 1) & (df_tel['is_true_pert'])]
    rho_t1_pert, p_t1_pert = spearmanr(df_t1_pert['claimed_impact'], df_t1_pert['abs_delta_days'])
    dir_fid_t1_pert = float(df_t1_pert['dir_match'].mean() * 100)
    print(f"Top-1 Spearman rho on True Perturbations (N={len(df_t1_pert)}): {rho_t1_pert:.4f} (p = {p_t1_pert:.4e})")
    print(f"Top-1 Directional Fidelity on True Perturbations (N={len(df_t1_pert)}): {dir_fid_t1_pert:.1f}%")

    # Non-zero deltas
    df_nz = df_tel[df_tel['abs_delta_days'] > 1e-4]
    rho_nz, p_nz = spearmanr(df_nz['claimed_impact'], df_nz['abs_delta_days'])
    print(f"Top-3 Spearman rho on Non-Zero Deltas (N={len(df_nz)}): {rho_nz:.4f} (p = {p_nz:.4e})")

    results = {
        'total_trials': total_trials,
        'true_perturbations_count': n_pert,
        'top1_spearman_rho_all': float(rho_t1),
        'top1_spearman_pvalue_all': float(p_t1),
        'top3_spearman_rho_all': float(rho_t3),
        'top3_spearman_pvalue_all': float(p_t3),
        'top1_directional_fidelity_all_pct': float(np.mean(top1_dirs) * 100),
        'top3_directional_fidelity_all_pct': float(np.mean(top3_dirs) * 100),
        'top3_spearman_rho_true_pert': float(rho_pert),
        'top3_spearman_pvalue_true_pert': float(p_pert),
        'top3_directional_fidelity_true_pert_pct': dir_fid_pert,
        'top1_spearman_rho_true_pert': float(rho_t1_pert),
        'top1_spearman_pvalue_true_pert': float(p_t1_pert),
        'top1_directional_fidelity_true_pert_pct': dir_fid_t1_pert,
        'top3_spearman_rho_non_zero': float(rho_nz),
        'top3_spearman_pvalue_non_zero': float(p_nz),
        'mean_top1_delta_days': float(np.mean(top1_measured)),
        'median_top1_delta_days': float(np.median(top1_measured)),
        'telemetry': telemetry
    }

    with open('timeline_faithfulness_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nTIMELINE FAITHFULNESS AUDIT COMPLETE -> Saved to timeline_faithfulness_results.json")

if __name__ == '__main__':
    run_timeline_audit()
