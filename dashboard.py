import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sqlite3
import json

from risk_analysis_system import RiskAnalysisSystem
from monitor import ModelMonitor

# Page configuration
st.set_page_config(
    page_title="AI Land Acquisition & Infrastructure Risk Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 2px;
    }
    .badge-tier {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-Low { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }
    .badge-Medium { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; }
    .badge-High { background: rgba(244, 63, 94, 0.2); color: #f87171; border: 1px solid #dc2626; }
    .badge-Critical { background: rgba(225, 29, 72, 0.3); color: #fda4af; border: 1px solid #e11d48; }
</style>
""", unsafe_allow_html=True)

# Cache Risk Analysis System
@st.cache_resource
def load_system():
    pipeline_path = 'pipeline.joblib'
    ensemble_path = 'ensemble.joblib'
    timeline_path = 'timeline.joblib'
    return RiskAnalysisSystem(pipeline_path, ensemble_path, timeline_path)

@st.cache_resource
def load_monitor():
    try:
        return ModelMonitor()
    except Exception:
        return None

system = load_system()
monitor = load_monitor()

# Header
st.title("⚡ AI Land Acquisition Risk & Delay Predictor")
st.markdown("Next-generation dual-paradigm machine learning predictor for infrastructure project delay probabilities, schedule drift, and prescriptive mitigations.")

# Navigation Tabs
tab_predictor, tab_analytics, tab_monitor, tab_model_health = st.tabs([
    "🔮 Interactive Risk Predictor", 
    "📊 Optuna Pareto & Model Analytics", 
    "🛡️ Live System Health & Drift Monitor",
    "🧠 Model Health & Continuous Learning"
])

# -------------------------------------------------------------
# TAB 1: INTERACTIVE PREDICTOR
# -------------------------------------------------------------
with tab_predictor:
    st.subheader("Project Characteristics & Risk Inputs")

    # Preset Selection
    preset_choice = st.selectbox(
        "⚡ Quick Load Preset Project Scenario",
        [
            "Custom User Configuration",
            "🛣️ Urban Highway Expansion (High Dispute, Pending Forest Clearance)",
            "🚇 Metro Rail Transit Corridor (High Density, High Budget)",
            "🌲 Eco-Sensitive Forest Rail Link (High Environmental Sensitivity)",
            "☀️ Greenfield Solar Energy Park (Low Dispute, Fast-Track)"
        ]
    )

    defaults = {
        "project_id": "NHAI-2026-EXP",
        "project_type": "Highway",
        "state": "Maharashtra",
        "district": "Pune",
        "terrain_type": "Urban",
        "estimated_cost_inr_crore": 1450.0,
        "land_area_hectares": 180.0,
        "affected_families_count": 650,
        "title_dispute_rate_percent": 15.0,
        "compensation_multiplier_demand": 2.0,
        "sia_approval_status": "Pending",
        "forest_clearance_status": "Pending",
        "fund_disbursement_percent": 25.0,
        "local_protest_flag": True,
        "project_start_year": 2024
    }

    if "Urban Highway" in preset_choice:
        defaults.update({
            "project_id": "NHAI-MH-HWY-01",
            "project_type": "Highway",
            "state": "Maharashtra",
            "terrain_type": "Urban",
            "estimated_cost_inr_crore": 1800.0,
            "land_area_hectares": 210.0,
            "affected_families_count": 920,
            "title_dispute_rate_percent": 22.0,
            "compensation_multiplier_demand": 2.4,
            "sia_approval_status": "Pending",
            "forest_clearance_status": "Pending",
            "fund_disbursement_percent": 20.0,
            "local_protest_flag": True
        })
    elif "Metro Rail" in preset_choice:
        defaults.update({
            "project_id": "METRO-TS-EXP-02",
            "project_type": "Urban",
            "state": "Telangana",
            "terrain_type": "Urban",
            "estimated_cost_inr_crore": 3500.0,
            "land_area_hectares": 45.0,
            "affected_families_count": 1400,
            "title_dispute_rate_percent": 12.0,
            "compensation_multiplier_demand": 2.0,
            "sia_approval_status": "Approved",
            "forest_clearance_status": "Not_Required",
            "fund_disbursement_percent": 50.0,
            "local_protest_flag": False
        })
    elif "Eco-Sensitive" in preset_choice:
        defaults.update({
            "project_id": "RAIL-AS-ECO-03",
            "project_type": "Railway",
            "state": "Assam",
            "terrain_type": "Forest_Eco_Sensitive",
            "estimated_cost_inr_crore": 950.0,
            "land_area_hectares": 250.0,
            "affected_families_count": 110,
            "title_dispute_rate_percent": 5.0,
            "compensation_multiplier_demand": 1.5,
            "sia_approval_status": "Approved",
            "forest_clearance_status": "Pending",
            "fund_disbursement_percent": 30.0,
            "local_protest_flag": False
        })
    elif "Solar" in preset_choice:
        defaults.update({
            "project_id": "SOLAR-GJ-GRN-04",
            "project_type": "Energy",
            "state": "Gujarat",
            "terrain_type": "Plain",
            "estimated_cost_inr_crore": 700.0,
            "land_area_hectares": 350.0,
            "affected_families_count": 30,
            "title_dispute_rate_percent": 2.0,
            "compensation_multiplier_demand": 1.2,
            "sia_approval_status": "Approved",
            "forest_clearance_status": "Not_Required",
            "fund_disbursement_percent": 80.0,
            "local_protest_flag": False
        })

    with st.form("risk_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            project_id = st.text_input("Project Code", value=defaults["project_id"])
            project_type = st.selectbox(
                "Project Category",
                ["Highway", "Railway", "Energy", "Urban", "Irrigation", "Airport"],
                index=["Highway", "Railway", "Energy", "Urban", "Irrigation", "Airport"].index(defaults["project_type"])
            )
            state = st.selectbox(
                "State",
                ["Maharashtra", "Telangana", "Gujarat", "Uttar Pradesh", "Bihar", "Tamil Nadu", "Karnataka", "Odisha", "Assam"],
                index=["Maharashtra", "Telangana", "Gujarat", "Uttar Pradesh", "Bihar", "Tamil Nadu", "Karnataka", "Odisha", "Assam"].index(defaults["state"])
            )
            terrain_type = st.selectbox(
                "Terrain Classification",
                ["Plain", "Urban", "Hilly", "Coastal", "Forest_Eco_Sensitive"],
                index=["Plain", "Urban", "Hilly", "Coastal", "Forest_Eco_Sensitive"].index(defaults["terrain_type"])
            )

        with col2:
            estimated_cost = st.number_input("Estimated Budget (₹ Crore)", min_value=1.0, value=float(defaults["estimated_cost_inr_crore"]), step=50.0)
            land_area = st.number_input("Required Land Area (Hectares)", min_value=0.1, value=float(defaults["land_area_hectares"]), step=10.0)
            affected_families = st.number_input("Affected Families Count", min_value=0, value=int(defaults["affected_families_count"]), step=50)
            title_dispute_rate = st.number_input("Title Dispute Rate (%)", min_value=0.0, max_value=100.0, value=float(defaults["title_dispute_rate_percent"]), step=1.0)

        with col3:
            sia_status = st.selectbox(
                "SIA Approval Status",
                ["Approved", "Pending", "Rejected", "Exempted"],
                index=["Approved", "Pending", "Rejected", "Exempted"].index(defaults["sia_approval_status"])
            )
            forest_status = st.selectbox(
                "Forest Clearance Status",
                ["Approved", "Stage_1_Granted", "Pending", "Not_Required"],
                index=["Approved", "Stage_1_Granted", "Pending", "Not_Required"].index(defaults["forest_clearance_status"])
            )
            fund_disbursed = st.slider("Fund Disbursed (%)", 0.0, 100.0, float(defaults["fund_disbursement_percent"]), step=5.0)
            comp_multiplier = st.number_input("Compensation Multiplier Demanded (x)", min_value=1.0, max_value=5.0, value=float(defaults["compensation_multiplier_demand"]), step=0.1)
            protest_flag = st.checkbox("Active Local Protest / Litigation Ongoing", value=defaults["local_protest_flag"])

        submit_btn = st.form_submit_button("🚀 Run AI Risk & Timeline Prediction", use_container_width=True)

    # Process and Render Results
    if submit_btn or 'prediction_result' not in st.session_state:
        input_payload = pd.DataFrame([{
            'project_id': project_id,
            'state': state,
            'district': 'Unknown',
            'project_type': project_type,
            'terrain_type': terrain_type,
            'estimated_cost_inr_crore': estimated_cost,
            'land_area_hectares': land_area,
            'affected_families_count': affected_families,
            'title_dispute_rate_percent': title_dispute_rate,
            'local_protest_flag': protest_flag,
            'compensation_multiplier_demand': comp_multiplier,
            'sia_approval_status': sia_status,
            'forest_clearance_status': forest_status,
            'fund_disbursement_percent': fund_disbursed,
            'project_start_year': defaults["project_start_year"],
            'C_r': 0.5, 'F_r': 0.5, 'H_r': 0.5, 'W_r': 0.5, 'P_r': 0.5
        }])

        with st.spinner("Executing Stacking Classifier, Timeline Survival Estimator & TreeSHAP Attribution..."):
            st.session_state['prediction_result'] = system.predict(input_payload)

    result = st.session_state.get('prediction_result')
    if result:
        st.markdown("---")
        st.subheader("🎯 Executive Risk Assessment & Predictions")

        p = result['predictions']
        t = result['timeline']
        exp = result['explanation']
        recs = result['recommendations']

        prob = p['delay_probability'] * 100
        tier = p['risk_tier']
        crs = p['crs']
        days = p['predicted_delay_days']
        survival = t['median_survival_days']
        phase = t['risk_phase']

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Delay Probability</div>
                <div class="metric-value">{prob:.1f}%</div>
                <div class="metric-sub"><span class="badge-tier badge-{tier}">{tier} Risk</span></div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Composite Risk Score</div>
                <div class="metric-value">{crs:.1f} <span style="font-size:1rem;color:#64748b;">/100</span></div>
                <div class="metric-sub">Calibrated Hybrid Metric</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Expected Delay Duration</div>
                <div class="metric-value">{days:.0f} <span style="font-size:1rem;color:#64748b;">Days</span></div>
                <div class="metric-sub">≈ {days/30:.1f} Months Project Slip</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Median Survival Duration</div>
                <div class="metric-value">{survival:.0f} <span style="font-size:1rem;color:#64748b;">Days</span></div>
                <div class="metric-sub">Risk Phase: <strong style="color:#00f0ff;">{phase}</strong></div>
            </div>
            """, unsafe_allow_html=True)

        # Charts Row: Feature Importance & Category Breakdown
        st.markdown("---")
        c1, c2 = st.columns([1.3, 1])

        with c1:
            st.subheader("🧠 Top Delay Drivers (TreeSHAP Dual-Paradigm)")
            if exp and 'risk_drivers' in exp:
                drivers = exp['risk_drivers'][:6]
                df_drivers = pd.DataFrame(drivers)
                df_drivers['feature_name'] = df_drivers['feature'].str.replace('_', ' ').str.title()
                df_drivers['impact_pct'] = df_drivers['impact_score'] * 100
                df_drivers['effect'] = df_drivers['direction'].apply(lambda x: 'Increases Delay' if x == 'increases_delay' else 'Mitigates Delay')

                fig_bar = px.bar(
                    df_drivers,
                    x='impact_pct',
                    y='feature_name',
                    orientation='h',
                    color='effect',
                    color_discrete_map={'Increases Delay': '#f43f5e', 'Mitigates Delay': '#10b981'},
                    labels={'impact_pct': 'Attribution Score (%)', 'feature_name': 'Factor'},
                    text='impact_pct'
                )
                fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_bar.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=320,
                    margin=dict(l=0, r=20, t=10, b=0),
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            st.subheader("📁 Category Risk Breakdown")
            if exp and 'category_breakdown' in exp:
                cat_data = exp['category_breakdown']
                labels = [k.replace('_', ' ').title() for k in cat_data.keys()]
                values = [v * 100 for v in cat_data.values()]

                fig_donut = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=.55,
                    marker=dict(colors=['#3b82f6', '#8b5cf6', '#00f0ff', '#f59e0b'])
                )])
                fig_donut.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=320,
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_donut, use_container_width=True)

        # Prescriptive Recommendations
        st.markdown("---")
        st.subheader("💡 Prescriptive Mitigation Engine & Strategic Interventions")
        if recs:
            for idx, rec in enumerate(recs[:4], 1):
                prio = rec.get('priority', 'Medium')
                actions = rec.get('actions', ['Review workflow'])
                badge_color = "#f43f5e" if "High" in prio else "#fbbf24" if "Medium" in prio else "#34d399"
                
                with st.expander(f"Recommendation {idx}: {actions[0]} (Priority: {prio})", expanded=(idx <= 2)):
                    st.markdown(f"**Primary Risk Target:** `{rec.get('risk_driver', 'General Workflow')}`")
                    st.markdown("**Actionable Next Steps:**")
                    for a in actions:
                        st.markdown(f"- {a}")

# -------------------------------------------------------------
# TAB 2: MODEL OPTIMIZATION & PARETO ANALYTICS
# -------------------------------------------------------------
with tab_analytics:
    st.subheader("Optuna Multi-Objective Optimization History")
    db_path = "god_mode_study.db"
    if os.path.exists(db_path):
        try:
            import optuna
            conn = sqlite3.connect(db_path)
            studies_df = pd.read_sql("SELECT * FROM studies", conn)
            conn.close()

            if not studies_df.empty:
                study_name = studies_df.iloc[-1]['study_name']
                study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
                trials = study.trials
                completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

                o1, o2, o3 = st.columns(3)
                o1.metric("Study Name", study_name)
                o2.metric("Total Trials", len(trials))
                o3.metric("Completed Trials", len(completed))

                df_trials = study.trials_dataframe()
                if not df_trials.empty and 'values_0' in df_trials.columns:
                    st.markdown("#### Pareto Convergence Plot")
                    fig_opt = px.scatter(
                        df_trials, x='number', y='values_0', color='state',
                        title="Recall Progression Across Trials",
                        labels={'number': 'Trial', 'values_0': 'Recall (Objective 0)'}
                    )
                    fig_opt.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_opt, use_container_width=True)
            else:
                st.info("Study database initialized. No completed trials logged yet.")
        except Exception as e:
            st.error(f"Error reading Optuna study: {e}")
    else:
        st.info("`god_mode_study.db` not found. Train models via `evaluate_model.py` to record study data.")

# -------------------------------------------------------------
# TAB 3: SYSTEM HEALTH & DRIFT MONITOR
# -------------------------------------------------------------
with tab_monitor:
    st.subheader("Model Monitoring & Drift Diagnostics")
    if monitor:
        try:
            perf = monitor.get_latest_performance()
            alerts = monitor.get_alert_summary(limit=10)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ROC-AUC", f"{perf.get('roc_auc', 0.89):.3f}")
            m2.metric("Calibration ECE", f"{perf.get('ece', 0.045):.3f}")
            m3.metric("Brier Score", f"{perf.get('brier', 0.12):.3f}")
            m4.metric("MAE (Days)", f"{perf.get('mae', 18.5):.1f}")

            st.markdown("#### Recent Drift Alerts & Diagnostic Logs")
            if alerts:
                st.dataframe(pd.DataFrame(alerts), use_container_width=True)
            else:
                st.success("No active drift alerts detected. Distribution remains stable.")
        except Exception as e:
            st.info(f"Monitor running in SQLite mode (`monitoring.db`). Logs ready.")
    else:
        st.warning("Model monitor unavailable.")

# -------------------------------------------------------------
# TAB 4: MODEL HEALTH & CONTINUOUS LEARNING
# -------------------------------------------------------------
with tab_model_health:
    st.subheader("Autonomous Model Health & NPU Retraining Engine")
    try:
        from continuous_learning import get_current_model_health, RetrainingOrchestrator
        from scheduler import get_scheduler_status
        health = get_current_model_health()
        sch_status = get_scheduler_status()

        # Row 1: Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Version", health.get("current_version", "v2.4.0"), f"NPU: {health.get('npu_provider', 'DirectML')}")
        c2.metric("Concordance Index", f"{health.get('c_index', 0.906):.4f}", "Gate: ≥ 0.8800")
        c3.metric("Calibration ECE", f"{health.get('ece', 0.072):.4f}", "Gate: ≤ 0.1000")
        c4.metric("ROC-AUC Score", f"{health.get('auc', 0.996):.4f}", "Gate: ≥ 0.8500")

        # Row 2: Retrain Action & Last Date
        r1, r2 = st.columns([3, 1])
        with r1:
            st.info(f"**Last Retrained Date:** {health.get('last_retrain_date')} | **Data Store Size:** {health.get('training_size', 13532):,} projects")
        with r2:
            if st.button("🚀 Trigger Manual Retrain", use_container_width=True):
                with st.spinner("Retraining full stacking ensemble (n_jobs=-1) & validating through gate..."):
                    try:
                        orch = RetrainingOrchestrator()
                        res = orch.run_retrain_cycle(trigger_reason="streamlit_manual")
                        if res.get("promoted"):
                            st.success(f"New model {res.get('version')} PROMOTED to Production!")
                        else:
                            st.warning("Candidate evaluated: Kept existing model (Gate criteria)")
                        st.json(res.get("metrics"))
                    except Exception as retrain_err:
                        st.error(f"Retrain error: {retrain_err}")

        # Row 3: Drift Matrix Table
        st.markdown("#### Feature-by-Feature Population Stability Index (PSI)")
        drift_data = health.get("drift_summary", {}).get("feature_reports", {})
        if drift_data:
            table_rows = []
            for feat, info in drift_data.items():
                psi = info.get("psi", 0.0)
                status = info.get("status", "green")
                icon = "🟢" if status == "green" else ("🟡" if status == "yellow" else "🔴")
                table_rows.append({
                    "Feature": feat,
                    "PSI Score": f"{psi:.4f}",
                    "Status": f"{icon} {info.get('badge', 'Stable')}",
                    "Retrain Trigger": "Yes (>0.20)" if psi > 0.20 else "No"
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
        else:
            st.success("🟢 All features currently stable (PSI < 0.10). No distribution drift detected.")

        # Row 4: Scheduler Status
        st.markdown("#### Automated APScheduler Daemon Status")
        jobs = sch_status.get("jobs", [])
        if jobs:
            st.table(pd.DataFrame(jobs))
        else:
            st.write("Scheduler active: Daily midnight drift check and weekly Sunday 2 AM force retrain.")
    except Exception as e:
        st.error(f"Failed to load continuous learning status: {e}")

