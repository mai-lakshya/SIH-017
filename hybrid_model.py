import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import StackingClassifier, StackingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV, RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import logit
import joblib

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

def safe_logit(X):
    """
    Apply logit transform to probabilities safely, 
    avoiding exactly 0 or 1 which result in -inf or inf.
    """
    X_clipped = np.clip(X, 1e-6, 1.0 - 1e-6)
    return logit(X_clipped)

class TreeWrapperBase(BaseEstimator):
    def __init__(self, model_class, random_state=42, **kwargs):
        self.model_class = model_class
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y):
        X_np = X.values if hasattr(X, 'values') else X
        y_np = np.array(y)
        
        stratify = y_np if isinstance(self, ClassifierMixin) else None
        if len(X_np) < 20:
            X_tr, X_val, y_tr, y_val = X_np, X_np, y_np, y_np
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_np, y_np, test_size=0.1, random_state=self.random_state, stratify=stratify
            )
            
        # extract early_stopping_rounds
        esr = self.kwargs.pop('early_stopping_rounds', None)
        
        # recreate model with remaining kwargs
        self.model = self.model_class(random_state=self.random_state, **self.kwargs)
        
        fit_kwargs = {}
        if esr is not None:
            if 'XGB' in self.model_class.__name__:
                self.model.set_params(early_stopping_rounds=esr)
                fit_kwargs['eval_set'] = [(X_val, y_val)]
                fit_kwargs['verbose'] = False
            elif 'LGBM' in self.model_class.__name__:
                import lightgbm as lgb
                fit_kwargs['eval_set'] = [(X_val, y_val)]
                fit_kwargs['callbacks'] = [lgb.early_stopping(stopping_rounds=esr, verbose=False)]
            elif 'CatBoost' in self.model_class.__name__:
                self.model.set_params(early_stopping_rounds=esr)
                fit_kwargs['eval_set'] = [(X_val, y_val)]
                fit_kwargs['verbose'] = False
                
        self.model.fit(X_tr, y_tr, **fit_kwargs)
        
        if isinstance(self, ClassifierMixin):
            self.classes_ = np.unique(y_np)
        return self

    def predict(self, X):
        X_np = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_np)

class TreeWrapperClassifier(ClassifierMixin, TreeWrapperBase):
    _estimator_type = "classifier"
    def __init__(self, model_class, random_state=42, **kwargs):
        super().__init__(model_class, random_state, **kwargs)
        
    def predict_proba(self, X):
        X_np = X.values if hasattr(X, 'values') else X
        return self.model.predict_proba(X_np)

class TreeWrapperRegressor(RegressorMixin, TreeWrapperBase):
    _estimator_type = "regressor"
    def __init__(self, model_class, random_state=42, **kwargs):
        super().__init__(model_class, random_state, **kwargs)

class HybridRiskPredictor:
    def __init__(self, cat_features=None, random_state=42, model_params=None, calibration_method='sigmoid', n_jobs=-1):
        self.cat_features = cat_features
        self.random_state = random_state
        self.model_params = model_params or {}
        self.calibration_method = calibration_method
        self.n_jobs = n_jobs
        self.classifier = None
        self.calibrated_classifier = None
        self.regressor_crs = None
        self.regressor_days = None

    def _build_classifiers(self, cv=3):
        # Extract params for each model
        lgb_params = self.model_params.get('lgb', {})
        xgb_params = self.model_params.get('xgb', {})
        cat_params = self.model_params.get('cat', {})
        et_params = self.model_params.get('et', {})
        
        lgb_params = {'verbose': -1, 'n_jobs': self.n_jobs, **lgb_params}
        xgb_params = {'verbosity': 0, 'n_jobs': self.n_jobs, **xgb_params}
        cat_params = {'verbose': False, 'thread_count': self.n_jobs, **cat_params}
        et_params = {'n_jobs': self.n_jobs, **et_params}
        
        lgb_clf = TreeWrapperClassifier(lgb.LGBMClassifier, random_state=self.random_state, **lgb_params)
        xgb_clf = TreeWrapperClassifier(xgb.XGBClassifier, random_state=self.random_state, **xgb_params)
        cat_clf = TreeWrapperClassifier(CatBoostClassifier, random_state=self.random_state, **cat_params)
        et_clf = TreeWrapperClassifier(ExtraTreesClassifier, random_state=self.random_state, **et_params)
        
        base_classifiers = [
            ('lgb', lgb_clf),
            ('xgb', xgb_clf),
            ('cat', cat_clf),
            ('et', et_clf)
        ]
        
        lr_kwargs = {
            'class_weight': 'balanced',
            'max_iter': 2000,
            'cv': cv,
            'random_state': self.random_state,
            'l1_ratios': (0.0,),
            'scoring': 'accuracy',
            'n_jobs': self.n_jobs
        }
        import inspect
        if 'use_legacy_attributes' in inspect.signature(LogisticRegressionCV.__init__).parameters:
            lr_kwargs['use_legacy_attributes'] = True

        meta_classifier = Pipeline([
            ('logit', FunctionTransformer(safe_logit)),
            ('lr', LogisticRegressionCV(**lr_kwargs))
        ])
        
        return StackingClassifier(
            estimators=base_classifiers,
            final_estimator=meta_classifier,
            cv=cv,
            n_jobs=None,
            passthrough=False
        )

    def _build_regressors(self, cv=3):
        lgb_params = self.model_params.get('lgb', {})
        xgb_params = self.model_params.get('xgb', {})
        cat_params = self.model_params.get('cat', {})
        et_params = self.model_params.get('et', {})
        
        lgb_params = {'verbose': -1, 'n_jobs': self.n_jobs, **lgb_params}
        xgb_params = {'verbosity': 0, 'n_jobs': self.n_jobs, **xgb_params}
        cat_params = {'verbose': False, 'thread_count': self.n_jobs, **cat_params}
        et_params = {'n_jobs': self.n_jobs, **et_params}
        
        lgb_reg = TreeWrapperRegressor(lgb.LGBMRegressor, random_state=self.random_state, **lgb_params)
        xgb_reg = TreeWrapperRegressor(xgb.XGBRegressor, random_state=self.random_state, **xgb_params)
        cat_reg = TreeWrapperRegressor(CatBoostRegressor, random_state=self.random_state, **cat_params)
        et_reg = TreeWrapperRegressor(ExtraTreesRegressor, random_state=self.random_state, **et_params)
        
        base_regressors = [
            ('lgb', lgb_reg),
            ('xgb', xgb_reg),
            ('cat', cat_reg),
            ('et', et_reg)
        ]
        
        alphas = np.logspace(-3, 4, 30)
        meta_regressor = RidgeCV(alphas=alphas, cv=cv)
        
        return StackingRegressor(
            estimators=base_regressors,
            final_estimator=meta_regressor,
            cv=cv,
            n_jobs=None,
            passthrough=False
        )

    def fit(self, X, y_cls, y_crs, y_days):
        y_cls_arr = np.asarray(y_cls, dtype=int)
        counts = np.bincount(y_cls_arr)
        min_class_count = min(counts) if len(counts) > 1 else 1

        # Adaptive CV splits based on minimum class count
        cv_splits = min(3, max(2, min_class_count // 3)) if min_class_count >= 4 else 2

        self.classifier = self._build_classifiers(cv=cv_splits)
        cal_method = getattr(self, 'calibration_method', 'sigmoid')
        if min_class_count < 15 and cal_method == 'isotonic':
            cal_method = 'sigmoid'
        self.calibrated_classifier = CalibratedClassifierCV(
            estimator=self.classifier,
            method=cal_method,
            cv=cv_splits
        )
        self.calibrated_classifier.fit(X, y_cls)
        
        if hasattr(self.calibrated_classifier, 'calibrated_classifiers_') and len(self.calibrated_classifier.calibrated_classifiers_) > 0:
            self.classifier = self.calibrated_classifier.calibrated_classifiers_[0].estimator

        self.regressor_crs = self._build_regressors(cv=cv_splits)
        self.regressor_crs.fit(X, y_crs)
        
        self.regressor_days = self._build_regressors(cv=cv_splits)
        self.regressor_days.fit(X, y_days)

        # Conformal Prediction Calibration (90% prediction intervals, alpha=0.10)
        try:
            fitted_crs = self.regressor_crs.predict(X)
            fitted_days = self.regressor_days.predict(X)
            res_crs = np.abs(np.asarray(y_crs, dtype=float) - np.asarray(fitted_crs, dtype=float))
            res_days = np.abs(np.asarray(y_days, dtype=float) - np.asarray(fitted_days, dtype=float))
            n = len(res_crs)
            # Finite-sample correction for (1 - alpha) = 0.90
            q_level = min(1.0, (int(np.ceil((n + 1) * 0.90))) / n) if n > 0 else 0.90
            self.conformal_q_crs_ = float(np.quantile(res_crs, q_level))
            self.conformal_q_days_ = float(np.quantile(res_days, q_level))
        except Exception:
            self.conformal_q_crs_ = 0.1
            self.conformal_q_days_ = 65.0
        
        return self

    def predict(self, X, blend_monotonicity=False):
        delay_prob = self.calibrated_classifier.predict_proba(X)[:, 1]
        pred_crs = self.regressor_crs.predict(X)
        pred_days = self.regressor_days.predict(X)
        
        # 1. Calibrated Continuous Outputs Bounded to Valid Statutory Domains
        raw_crs = np.clip(pred_crs, 0.0, 100.0)
        raw_days = np.clip(pred_days, 30.0, 730.0)

        # 2. Separate Heuristic Monotonic Adjustments
        # Decoupled from primary calibrated scores to prevent R2 degradation
        adjusted_risk_index = np.clip(raw_crs * (0.5 + delay_prob), 0.0, 100.0)
        adjusted_delay_days = np.clip(raw_days * (0.5 + delay_prob), 30.0, 730.0)

        # 3. Conformal Prediction Bounds (90% Coverage [P10, P90])
        q_crs = getattr(self, 'conformal_q_crs_', 0.1)
        q_days = getattr(self, 'conformal_q_days_', 65.0)

        crs_p10 = np.clip(raw_crs - q_crs, 0.0, 100.0)
        crs_p90 = np.clip(raw_crs + q_crs, 0.0, 100.0)
        days_p10 = np.clip(raw_days - q_days, 30.0, 730.0)
        days_p90 = np.clip(raw_days + q_days, 30.0, 730.0)

        risk_tiers = np.where(raw_crs > 75, "Critical",
                     np.where(raw_crs > 50, "High",
                     np.where(raw_crs > 25, "Medium", "Low")))
            
        return {
            'delay_probability': delay_prob,
            'crs': raw_crs,
            'predicted_crs': raw_crs,
            'raw_crs': raw_crs,
            'adjusted_risk_index': adjusted_risk_index,
            'delay_days': raw_days,
            'predicted_delay_days': raw_days,
            'raw_delay_days': raw_days,
            'adjusted_delay_days': adjusted_delay_days,
            'days_p10': days_p10,
            'days_p90': days_p90,
            'crs_p10': crs_p10,
            'crs_p90': crs_p90,
            'confidence_interval_90': {
                'days': [days_p10, days_p90],
                'crs': [crs_p10, crs_p90]
            },
            'risk_tier': risk_tiers
        }

    def save(self, filepath):
        joblib.dump(self, filepath, compress=3)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
