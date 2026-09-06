"""
Timeline Permutation Explainer for NonLinearTimelinePredictor (RandomSurvivalForest).
Computes feature importance using Uno's IPCW C-index permutation degradation,
providing feature-level rationales for predicted delay times and survival risks.
"""

import warnings
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored
from sksurv.util import Surv


class TimelinePermutationExplainer:
    """
    Model-agnostic permutation importance explainer tailored for sksurv's
    RandomSurvivalForest, which lacks native TreeSHAP and feature_importances_.
    Measures empirical degradation in Uno's C-index when each feature is
    randomly permuted on a reference background survival dataset.
    """

    def __init__(
        self,
        timeline_predictor,
        feature_names,
        background_data=None,
        background_events=None,
        background_times=None,
        n_repeats=3,
        random_state=42
    ):
        self.timeline_predictor = timeline_predictor
        self.feature_names = list(feature_names)
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.feature_importances_ = None
        self._cached_importance_list = None

        if hasattr(timeline_predictor, 'rsf'):
            self.rsf_model = timeline_predictor.rsf
        elif hasattr(timeline_predictor, 'predict') and not hasattr(timeline_predictor, 'predict_time_to_delay'):
            self.rsf_model = timeline_predictor
        else:
            self.rsf_model = getattr(timeline_predictor, 'rsf', None)

        if background_data is not None:
            bg_df = background_data if isinstance(background_data, pd.DataFrame) else pd.DataFrame(background_data, columns=self.feature_names)
            self.background_medians = bg_df.median(numeric_only=True).to_dict()
        else:
            self.background_medians = {}

        if background_data is not None and background_events is not None and background_times is not None:
            self.fit(background_data, background_events, background_times)

    @classmethod
    def from_rsf(
        cls,
        rsf_model,
        feature_names,
        background_data=None,
        background_events=None,
        background_times=None,
        n_repeats=3,
        random_state=42
    ):
        """
        Construct a TimelinePermutationExplainer directly from a bare RandomSurvivalForest
        estimator, decoupling permutation explanations from the full NonLinearTimelinePredictor
        container and its PyTorch/DeepSurv dependencies.
        """
        class _RSFWrapper:
            def __init__(self, m):
                self.rsf = m

            def predict(self, X):
                from survival_features import prepare_survival_features
                expected = getattr(self.rsf, 'feature_names_in_', None)
                X_prep = prepare_survival_features(X, expected)
                return self.rsf.predict(X_prep)

            def predict_survival_function(self, X):
                from survival_features import prepare_survival_features
                expected = getattr(self.rsf, 'feature_names_in_', None)
                X_prep = prepare_survival_features(X, expected)
                return self.rsf.predict_survival_function(X_prep)

        return cls(
            timeline_predictor=_RSFWrapper(rsf_model),
            feature_names=feature_names,
            background_data=background_data,
            background_events=background_events,
            background_times=background_times,
            n_repeats=n_repeats,
            random_state=random_state
        )

    def _predict_rsf(self, X):
        from survival_features import prepare_survival_features
        if self.rsf_model is not None and hasattr(self.rsf_model, 'predict'):
            expected = getattr(self.rsf_model, 'feature_names_in_', None)
            X_prep = prepare_survival_features(X, expected)
            return self.rsf_model.predict(X_prep)
        elif hasattr(self.timeline_predictor, 'predict'):
            return self.timeline_predictor.predict(X)
        return np.zeros(len(X), dtype=float)

    def fit(self, X_bg, events, times):
        """
        Computes permutation importance using Uno's C-index across features.
        
        NOTE ON IPCW ESTIMATION:
        concordance_index_ipcw uses the reference background set (y_surv) to estimate the
        Kaplan-Meier censoring distribution G(t) for inverse probability of censoring weights.
        When a separate hold-out censoring cohort is unavailable, evaluating permutation
        degradation on the reference set provides an unbiased relative ranking of feature sensitivity.
        """
        rng = np.random.RandomState(self.random_state)
        X_df = X_bg.copy() if isinstance(X_bg, pd.DataFrame) else pd.DataFrame(X_bg, columns=self.feature_names)

        # Update background medians from fitted background
        self.background_medians = X_df.median(numeric_only=True).to_dict()

        # Ensure column alignment
        for col in self.feature_names:
            if col not in X_df.columns:
                X_df[col] = 0.0
        X_df = X_df[self.feature_names].copy()

        events = np.asarray(events, dtype=bool)
        times = np.asarray(times, dtype=np.float64)
        times = np.maximum(times, 1.0)
        y_surv = Surv.from_arrays(event=events, time=times)

        if self.rsf_model is None:
            # Equal importance fallback if no RSF model
            self.feature_importances_ = {col: 1.0 / len(self.feature_names) for col in self.feature_names}
            self._build_cached_list()
            return self

        # Base risk prediction
        base_preds = self._predict_rsf(X_df)
        try:
            c_base, _, _, _, _ = concordance_index_ipcw(y_surv, y_surv, base_preds)
        except Exception:
            c_base, _, _, _, _ = concordance_index_censored(events, times, base_preds)

        raw_importances = {col: 0.0 for col in self.feature_names}

        for repeat in range(self.n_repeats):
            for col in self.feature_names:
                X_perm = X_df.copy()
                X_perm[col] = rng.permutation(X_perm[col].values)
                perm_preds = self._predict_rsf(X_perm)
                try:
                    c_perm, _, _, _, _ = concordance_index_ipcw(y_surv, y_surv, perm_preds)
                except Exception:
                    c_perm, _, _, _, _ = concordance_index_censored(events, times, perm_preds)

                # Feature importance is sensitivity/degradation in discrimination
                raw_importances[col] += abs(c_base - c_perm)

        # Average and normalize
        total = sum(raw_importances.values())
        if total > 1e-12:
            self.feature_importances_ = {col: float(val / total) for col, val in raw_importances.items()}
        else:
            self.feature_importances_ = {col: float(1.0 / len(self.feature_names)) for col in self.feature_names}

        self._build_cached_list()
        return self

    def _build_cached_list(self):
        sorted_items = sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)
        self._cached_importance_list = [
            {"feature": feat, "importance": float(imp)}
            for feat, imp in sorted_items
        ]

    def _predict_raw(self, X_df):
        if hasattr(self.timeline_predictor, 'predict_time_to_delay'):
            return np.asarray(self.timeline_predictor.predict_time_to_delay(X_df), dtype=float)
        elif self.rsf_model is not None and hasattr(self.rsf_model, 'predict'):
            from survival_features import prepare_survival_features
            expected = getattr(self.rsf_model, 'feature_names_in_', None)
            X_prep = prepare_survival_features(X_df, expected)
            return np.asarray(self.rsf_model.predict(X_prep), dtype=float)
        elif hasattr(self.timeline_predictor, 'predict'):
            res = self.timeline_predictor.predict(X_df)
            if isinstance(res, dict) and 'predicted_delay_days' in res:
                return np.asarray(res['predicted_delay_days'], dtype=float)
            return np.asarray(res, dtype=float)
        return np.zeros(len(X_df), dtype=float)

    def explain(self, row, top_k=5, mode="global"):
        r"""
        Generates feature-level rationale for a single instance or batch row.
        
        Parameters:
        -----------
        row : pd.DataFrame or dict
            The instance to explain.
        top_k : int
            Number of top features to return.
        mode : str, default "global"
            - "global": Ranks features by globally-estimated Uno's C-index permutation importance.
              (Documented: Explains overall survival risk drivers across the portfolio).
            - "local": Ranks features by instance-level marginal perturbation sensitivity
              |predict(row) - predict(row \ {f})| using reference background medians.
        """
        if self._cached_importance_list is None:
            # Default equal importances if fit not called
            self.feature_importances_ = {col: 1.0 / len(self.feature_names) for col in self.feature_names}
            self._build_cached_list()

        row_df = row.copy() if isinstance(row, pd.DataFrame) else pd.DataFrame([row])
        for col in self.feature_names:
            if col not in row_df.columns:
                row_df[col] = self.background_medians.get(col, 0.0)
        row_df = row_df[self.feature_names]

        if mode == "local":
            base_val_pred = float(self._predict_raw(row_df)[0])
            local_impacts = []
            for feat in self.feature_names:
                orig_val = float(row_df[feat].values[0])
                neut_val = float(self.background_medians.get(feat, 0.0))
                row_pert = row_df.copy()
                row_pert[feat] = neut_val
                pert_pred = float(self._predict_raw(row_pert)[0])
                abs_impact = abs(base_val_pred - pert_pred)
                direction = "increases_delay" if base_val_pred >= pert_pred else "decreases_delay"
                local_impacts.append({
                    "feature": feat,
                    "importance": float(abs_impact),
                    "value": orig_val,
                    "direction": direction,
                    "global_importance": float(self.feature_importances_.get(feat, 0.0))
                })
            local_impacts.sort(key=lambda x: x["importance"], reverse=True)
            top_drivers = local_impacts[:top_k]
            driver_names = [d["feature"] for d in top_drivers[:3]]
            rationale_str = f"Instance timeline risk driven by {', '.join(driver_names)} based on local marginal perturbation."
        else:
            top_drivers = []
            for item in self._cached_importance_list[:top_k]:
                feat = item["feature"]
                val = float(row_df[feat].values[0]) if feat in row_df.columns else 0.0
                top_drivers.append({
                    "feature": feat,
                    "importance": float(item["importance"]),
                    "value": val
                })
            driver_names = [d["feature"] for d in top_drivers[:3]]
            rationale_str = f"Global timeline risk driven by {', '.join(driver_names)} based on Uno's C-index survival permutation."

        return {
            "mode": mode,
            "top_drivers": top_drivers,
            "feature_importance": self._cached_importance_list,
            "rationale": rationale_str
        }

    def evaluate_faithfulness(self, X_eval, top_k=1):
        """
        Deletion faithfulness test for timeline predictor:
        Verifies that neutralizing the top driver for an instance causes a measurable
        change in the predicted timeline survival score on true non-zero perturbations.
        """
        X_df = X_eval.copy() if isinstance(X_eval, pd.DataFrame) else pd.DataFrame(X_eval, columns=self.feature_names)
        results = []
        for i in range(len(X_df)):
            row = X_df.iloc[[i]].copy()
            base_score = float(self._predict_raw(row)[0])
            exp = self.explain(row, top_k=top_k, mode="global")
            top_feat = exp["top_drivers"][0]["feature"]
            orig_val = float(row[top_feat].values[0])
            neut_val = float(self.background_medians.get(top_feat, 0.0))

            row_del = row.copy()
            row_del[top_feat] = neut_val
            del_score = float(self._predict_raw(row_del)[0])
            delta = abs(base_score - del_score)
            is_pert = abs(orig_val - neut_val) > 1e-5
            results.append({
                "row_index": i,
                "top_feature": top_feat,
                "orig_value": orig_val,
                "neutral_value": neut_val,
                "delta": delta,
                "is_true_perturbation": is_pert,
                "has_effect": delta > 1e-6
            })
        true_perts = [r for r in results if r["is_true_perturbation"]]
        effect_rate = np.mean([r["has_effect"] for r in true_perts]) if true_perts else 0.0
        return {
            "sample_size": len(results),
            "true_perturbation_count": len(true_perts),
            "effect_rate_on_perturbations": float(effect_rate),
            "results": results
        }
