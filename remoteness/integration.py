"""
Integration & Feature Engineering Hook for Main Delay-Prediction Model.
Augments training/inference dataframes with remoteness features, performs VIF
multicollinearity checks, computes permutation importance, and generates SHAP explainability plots.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, List, Optional

from .remoteness_score import RemotenessEvaluator

logger = logging.getLogger(__name__)


class RemotenessFeatureEngineer:
    """
    Transforms raw tabular project data by deriving the 6 remoteness features.
    Provides multicollinearity (VIF) checks and SHAP explainability hooks.
    """

    def __init__(self, evaluator: Optional[RemotenessEvaluator] = None):
        self.evaluator = evaluator or RemotenessEvaluator()

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
        allow_online: bool = False
    ) -> pd.DataFrame:
        """
        Augments input dataframe with:
        - remoteness_score_normalized (0-1)
        - remoteness_delay_days (float)
        - settlement_tier (categorical)
        - road_connectivity_type (categorical)
        - terrain_type (categorical)
        - distance_to_nearest_settlement_km (float)
        """
        out_df = df.copy()

        scores = []
        delays = []
        tiers = []
        roads = []
        terrains = []
        distances = []

        total = len(out_df)
        if verbose:
            logger.info(f"Extracting remoteness features for {total} projects...")

        for idx, row in out_df.iterrows():
            lat = row.get("latitude", row.get("lat"))
            lon = row.get("longitude", row.get("lon"))
            dist_name = row.get("district")
            proj_type = row.get("project_type")
            road = row.get("road_type", row.get("road_connectivity_type"))
            terrain = row.get("terrain_type")

            # If lat/lon missing, derive from district if available
            if pd.isna(lat) or pd.isna(lon) or lat is None or lon is None:
                if dist_name and str(dist_name).strip() not in ["", "nan", "Unknown"]:
                    coords, _, _ = self.evaluator.geocoder.geocode_address(
                        f"{dist_name}, {row.get('state', '')}", allow_online=allow_online
                    )
                    if not coords:
                        coords, _, _ = self.evaluator.geocoder.geocode_address(
                            str(dist_name).strip(), allow_online=allow_online
                        )
                    if coords:
                        lat, lon = coords
                    else:
                        lat, lon = 20.5937, 78.9629  # National centroid
                else:
                    lat, lon = 20.5937, 78.9629

            res = self.evaluator.evaluate(
                lat=lat,
                lon=lon,
                project_type=proj_type,
                district=dist_name,
                provided_road_type=road,
                provided_terrain=terrain,
                allow_online=allow_online
            )

            scores.append(res["remoteness_score_normalized"])
            delays.append(res["remoteness_delay_days"])
            tiers.append(res["nearest_settlement"]["tier"])
            roads.append(res["road_connectivity"]["type"])
            terrains.append(res["terrain_type"])
            distances.append(res["nearest_settlement"]["distance_km"])

        out_df["remoteness_score_normalized"] = scores
        out_df["remoteness_delay_days"] = delays
        out_df["settlement_tier"] = tiers
        out_df["road_connectivity_type"] = roads
        out_df["terrain_type"] = terrains
        out_df["distance_to_nearest_settlement_km"] = distances

        return out_df

    def compute_vif(self, df: pd.DataFrame, continuous_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Computes Variance Inflation Factor (VIF) to ensure remoteness features
        do not introduce severe multicollinearity (rule of thumb: VIF < 5-10).
        """
        if continuous_cols is None:
            continuous_cols = [
                col for col in [
                    "remoteness_score_normalized",
                    "remoteness_delay_days",
                    "distance_to_nearest_settlement_km",
                    "land_area_hectares",
                    "estimated_cost_inr_crore",
                    "affected_families_count",
                    "title_dispute_rate_percent",
                    "compensation_multiplier_demand",
                    "fund_disbursement_percent"
                ]
                if col in df.columns
            ]

        clean_df = df[continuous_cols].dropna().select_dtypes(include=[np.number])
        if clean_df.empty or len(clean_df.columns) < 2:
            return pd.DataFrame()

        vif_data = []
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            from statsmodels.tools.tools import add_constant
            X = add_constant(clean_df)
            for i in range(1, X.shape[1]):
                col = X.columns[i]
                vif_val = variance_inflation_factor(X.values, i)
                vif_data.append({
                    "feature": col,
                    "VIF": round(float(vif_val), 2),
                    "multicollinearity_status": "Low" if vif_val < 5.0 else ("Moderate" if vif_val < 10.0 else "High")
                })
        except ImportError:
            from sklearn.linear_model import LinearRegression
            cols = list(clean_df.columns)
            for i, col in enumerate(cols):
                y = clean_df[col].values
                X_other = clean_df[[c for c in cols if c != col]].values
                if np.var(y) == 0:
                    r2 = 0.0
                else:
                    lr = LinearRegression().fit(X_other, y)
                    r2 = float(lr.score(X_other, y))
                vif_val = 1.0 / max(1e-5, (1.0 - r2))
                vif_data.append({
                    "feature": col,
                    "VIF": round(float(vif_val), 2),
                    "multicollinearity_status": "Low" if vif_val < 5.0 else ("Moderate" if vif_val < 10.0 else "High")
                })

        return pd.DataFrame(vif_data).sort_values(by="VIF", ascending=False)

    def explain_with_shap(
        self,
        model,
        X_df: pd.DataFrame,
        output_dir: str = "remoteness/artifacts"
    ) -> Dict[str, Any]:
        """
        Generates TreeSHAP summary, dependence, and waterfall metrics for the retrained model.
        """
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "shap_remoteness_summary.png")
        dependence_path = os.path.join(output_dir, "shap_remoteness_dependence.png")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)

            # Handle binary classification list of shap arrays
            if isinstance(shap_values, list) and len(shap_values) == 2:
                vals = shap_values[1]
            else:
                vals = shap_values

            # 1. Beeswarm / Summary Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(vals, X_df, show=False)
            plt.tight_layout()
            plt.savefig(summary_path, dpi=150)
            plt.close()

            # 2. Dependence Plot if remoteness_score_normalized is present
            if "remoteness_score_normalized" in X_df.columns:
                plt.figure(figsize=(8, 5))
                interact_feature = "road_connectivity_type" if "road_connectivity_type" in X_df.columns else None
                shap.dependence_plot(
                    "remoteness_score_normalized",
                    vals,
                    X_df,
                    interaction_index=interact_feature,
                    show=False
                )
                plt.tight_layout()
                plt.savefig(dependence_path, dpi=150)
                plt.close()

            mean_abs_shap = np.mean(np.abs(vals), axis=0)
            importance_series = pd.Series(mean_abs_shap, index=X_df.columns).sort_values(ascending=False)

            return {
                "summary_plot": summary_path,
                "dependence_plot": dependence_path,
                "top_features": importance_series.head(10).to_dict()
            }
        except Exception as e:
            logger.warning(f"SHAP explanation generation skipped: {e}")
            return {"error": str(e)}

    def compare_ablation(
        self,
        X_base: pd.DataFrame,
        X_augmented: pd.DataFrame,
        y_days: np.ndarray,
        y_binary: np.ndarray
    ) -> Dict[str, Any]:
        """
        Ablation study comparing performance metrics with vs. without remoteness features.
        """
        from sklearn.model_selection import KFold
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.metrics import mean_absolute_error, root_mean_squared_error, precision_score, recall_score

        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        def eval_fold(X, y_r, y_c):
            # Numeric only for quick ablation
            X_num = X.select_dtypes(include=[np.number]).fillna(0.0)
            maes, rmses, precs, recs = [], [], [], []
            for train_idx, val_idx in kf.split(X_num):
                X_tr, X_val = X_num.iloc[train_idx], X_num.iloc[val_idx]
                y_r_tr, y_r_val = y_r[train_idx], y_r[val_idx]
                y_c_tr, y_c_val = y_c[train_idx], y_c[val_idx]

                reg = GradientBoostingRegressor(n_estimators=40, random_state=42)
                reg.fit(X_tr, y_r_tr)
                preds_r = reg.predict(X_val)
                maes.append(mean_absolute_error(y_r_val, preds_r))
                rmses.append(root_mean_squared_error(y_r_val, preds_r))

                clf = GradientBoostingClassifier(n_estimators=40, random_state=42)
                clf.fit(X_tr, y_c_tr)
                preds_c = clf.predict(X_val)
                precs.append(precision_score(y_c_val, preds_c, zero_division=0))
                recs.append(recall_score(y_c_val, preds_c, zero_division=0))

            return {
                "mae": round(float(np.mean(maes)), 2),
                "rmse": round(float(np.mean(rmses)), 2),
                "precision": round(float(np.mean(precs)), 3),
                "recall": round(float(np.mean(recs)), 3)
            }

        base_metrics = eval_fold(X_base, y_days, y_binary)
        aug_metrics = eval_fold(X_augmented, y_days, y_binary)

        return {
            "baseline_without_remoteness": base_metrics,
            "augmented_with_remoteness": aug_metrics,
            "mae_improvement_days": round(base_metrics["mae"] - aug_metrics["mae"], 2),
            "recall_delta": round(aug_metrics["recall"] - base_metrics["recall"], 3)
        }
