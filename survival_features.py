"""
Survival feature engineering and alignment utilities.
Provides consistent preparation of the 11 engineered features and 34 selected features
required for the high-performance Random Survival Forest and Gradient Boosted Survival Analysis.
"""

import numpy as np
import pandas as pd

SELECTED_SURVIVAL_FEATURES = [
    'project_type', 'district', 'terrain_type', 'estimated_cost_inr_crore',
    'project_start_year', 'affected_families_count',
    'title_dispute_rate_percent', 'local_protest_flag',
    'compensation_multiplier_demand', 'sia_approval_status',
    'forest_clearance_status', 'fund_disbursement_percent',
    'sia_approval_status_risk_score', 'forest_clearance_status_risk_score',
    'C_r', 'F_r', 'H_r', 'W_r', 'P_r', 'project_age_years', 'financial_density',
    'population_density', 'financial_burn_rate_to_date',
    'interact_legal_x_comp_pending', 'interact_legal_x_incomplete_docs',
    'ratio_disbursed_to_total_comp', 'ratio_docs_submitted_to_required',
    'velocity_notif_to_approval', 'velocity_approval_to_comp',
    'velocity_comp_to_possession', 'stage_velocity_rate', 'district_delay_rate',
    'days_since_last_activity', 'activity_staleness_ratio'
]

BASE_PIPELINE_FEATURES = [
    'project_type', 'state', 'district', 'terrain_type',
    'land_area_hectares', 'estimated_cost_inr_crore', 'project_start_year',
    'affected_families_count', 'title_dispute_rate_percent', 'local_protest_flag',
    'compensation_multiplier_demand', 'sia_approval_status', 'forest_clearance_status',
    'fund_disbursement_percent', 'sia_approval_status_risk_score',
    'forest_clearance_status_risk_score', 'C_r', 'F_r', 'H_r', 'W_r', 'P_r',
    'land_area_log', 'project_age_years', 'financial_density', 'population_density',
    'financial_burn_rate_to_date', 'state_project_type'
]


def prepare_survival_features(X, expected_features=None):
    """
    Transforms any DataFrame or array (raw or from pipeline.joblib) into the 34 features
    required by the high-concordance Random Survival Forest model.
    """
    if expected_features is None:
        expected_features = SELECTED_SURVIVAL_FEATURES
    else:
        expected_features = list(expected_features)

    # Convert array to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        if hasattr(X, 'shape') and X.shape[1] == len(expected_features):
            X_df = pd.DataFrame(X, columns=expected_features)
            return X_df
        elif hasattr(X, 'shape') and X.shape[1] == len(BASE_PIPELINE_FEATURES):
            X_df = pd.DataFrame(X, columns=BASE_PIPELINE_FEATURES)
        else:
            X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()

    # If all expected features already present, just return ordered
    if all(col in X_df.columns for col in expected_features):
        return X_df[expected_features]

    # Helper to safely retrieve Series with fallback
    def _col_series(col: str, default: float) -> pd.Series:
        if col in X_df.columns:
            return pd.to_numeric(X_df[col], errors='coerce').fillna(default)
        return pd.Series(default, index=X_df.index, dtype=float)

    # Otherwise derive missing engineered features
    # 1. Interactions
    title_disp = _col_series('title_dispute_rate_percent', 0.0)
    legal_disputes = title_disp / 100.0

    fund_disp_pct = _col_series('fund_disbursement_percent', 50.0)
    fund_disp = fund_disp_pct / 100.0
    comp_pending = 1.0 - fund_disp

    sia_risk = _col_series('sia_approval_status_risk_score', 0.5)
    forest_risk = _col_series('forest_clearance_status_risk_score', 0.5)
    incomplete_docs = (sia_risk + forest_risk) / 2.0

    if 'interact_legal_x_comp_pending' not in X_df.columns:
        X_df['interact_legal_x_comp_pending'] = legal_disputes * comp_pending

    if 'interact_legal_x_incomplete_docs' not in X_df.columns:
        X_df['interact_legal_x_incomplete_docs'] = legal_disputes * incomplete_docs

    # 2. Ratios
    if 'ratio_disbursed_to_total_comp' not in X_df.columns:
        comp_demand = _col_series('compensation_multiplier_demand', 1.0)
        X_df['ratio_disbursed_to_total_comp'] = fund_disp / np.maximum(comp_demand, 0.5)

    if 'ratio_docs_submitted_to_required' not in X_df.columns:
        X_df['ratio_docs_submitted_to_required'] = 1.0 - incomplete_docs

    # 3. Stage Velocity
    if 'section_11_notification_days' in X_df.columns:
        sec11_days = pd.to_numeric(X_df['section_11_notification_days'], errors='coerce').fillna(365.0)
    else:
        age_yrs = _col_series('project_age_years', 2.0)
        sec11_days = np.maximum(100.0, age_yrs * 180.0)

    if 'velocity_notif_to_approval' not in X_df.columns:
        X_df['velocity_notif_to_approval'] = sec11_days * (1.0 - sia_risk)

    if 'velocity_approval_to_comp' not in X_df.columns:
        X_df['velocity_approval_to_comp'] = sec11_days * fund_disp

    if 'velocity_comp_to_possession' not in X_df.columns:
        X_df['velocity_comp_to_possession'] = np.maximum(0.0, sec11_days - 365.0)

    if 'stage_velocity_rate' not in X_df.columns:
        X_df['stage_velocity_rate'] = (3.0 - incomplete_docs * 2.0 + fund_disp) / np.maximum(sec11_days / 100.0, 0.5)

    # 4. District Delay Rate
    if 'district_delay_rate' not in X_df.columns:
        if 'district' in X_df.columns and pd.api.types.is_numeric_dtype(X_df['district']):
            X_df['district_delay_rate'] = X_df['district']
        else:
            X_df['district_delay_rate'] = 0.55

    # 5. Days since last activity (staleness)
    age_yrs = _col_series('project_age_years', 2.0)
    proj_age_days = age_yrs * 365.25
    if 'days_since_last_activity' not in X_df.columns:
        X_df['days_since_last_activity'] = np.maximum(0.0, proj_age_days - sec11_days)

    if 'activity_staleness_ratio' not in X_df.columns:
        X_df['activity_staleness_ratio'] = X_df['days_since_last_activity'] / np.maximum(proj_age_days, 30.0)

    # Ensure all expected columns exist
    for col in expected_features:
        if col not in X_df.columns:
            X_df[col] = 0.0

    return X_df[expected_features]
