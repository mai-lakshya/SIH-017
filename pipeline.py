import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import KFold
import warnings

class DynamicFeatureTracker(BaseEstimator, TransformerMixin):
    """
    Automatically tracks categorical and continuous features to maintain
    consistent order and identify indices for downstream steps like SMOTE-NC.
    """
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols
        self.cat_cols_ = []
        self.cont_cols_ = []
        self.cat_indices_ = []

    def fit(self, X, y=None):
        self.feature_names_in_ = [c for c in X.columns if c not in ['project_id', 'project_index']]
        if self.cat_cols is None:
            # Dynamically identify categorical columns
            self.cat_cols_ = list(X.select_dtypes(include=['object', 'category', 'bool']).columns)
        else:
            self.cat_cols_ = [c for c in self.cat_cols if c in self.feature_names_in_]
            
        self.cont_cols_ = [c for c in self.feature_names_in_ if c not in self.cat_cols_]
        self.cat_indices_ = [self.feature_names_in_.index(c) for c in self.cat_cols_]
        return self

    def transform(self, X, y=None):
        # Ensure column order consistency and fill missing expected features
        if hasattr(self, 'feature_names_in_'):
            X_out = X.copy()
            for c in self.feature_names_in_:
                if c not in X_out.columns:
                    X_out[c] = "Unknown" if c in getattr(self, 'cat_cols_', []) else 0.0
                else:
                    if c in getattr(self, 'cat_cols_', []):
                        if pd.api.types.is_bool_dtype(X_out[c]):
                            if X_out[c].isna().any():
                                X_out[c] = X_out[c].astype(object).fillna("Unknown")
                        else:
                            X_out[c] = X_out[c].fillna("Unknown")
                    else:
                        X_out[c] = pd.to_numeric(X_out[c], errors='coerce').fillna(0.0)
            return X_out[self.feature_names_in_]
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Computes financial density, population density, velocity, and categorical interactions.
    """
    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = [c for c in X.columns if c not in ['project_id', 'project_index']]
        return self
        
    def transform(self, X, y=None):
        X_out = X.drop(columns=['project_id', 'project_index'], errors='ignore').copy()
        
        # 1. Ratios (Density)
        if 'estimated_cost_inr_crore' in X_out.columns and 'land_area_hectares' in X_out.columns:
            cost = pd.to_numeric(X_out['estimated_cost_inr_crore'], errors='coerce').fillna(0.0)
            area = pd.to_numeric(X_out['land_area_hectares'], errors='coerce').fillna(0.0)
            X_out['financial_density'] = cost / np.maximum(area, 1e-5)
            
        if 'affected_families_count' in X_out.columns and 'land_area_hectares' in X_out.columns:
            aff = pd.to_numeric(X_out['affected_families_count'], errors='coerce').fillna(0.0)
            area = pd.to_numeric(X_out['land_area_hectares'], errors='coerce').fillna(0.0)
            X_out['population_density'] = aff / np.maximum(area, 1e-5)
            
        # 2. Velocity (Fallback: Financial Burn Rate to date since planned_duration is missing)
        if 'estimated_cost_inr_crore' in X_out.columns and 'project_age_years' in X_out.columns:
            cost = pd.to_numeric(X_out['estimated_cost_inr_crore'], errors='coerce').fillna(0.0)
            age = pd.to_numeric(X_out['project_age_years'], errors='coerce').fillna(1.0)
            X_out['financial_burn_rate_to_date'] = cost / np.maximum(age, 1)
            
        # Temporal calculation: project_age_years from project_start_year if not present
        if 'project_start_year' in X_out.columns:
            start_year = pd.to_numeric(X_out['project_start_year'], errors='coerce')
            computed_age = np.maximum(0, 2026 - start_year)
            if 'project_age_years' not in X_out.columns:
                X_out['project_age_years'] = computed_age
            else:
                X_out['project_age_years'] = X_out['project_age_years'].fillna(computed_age)

        # Outlier clipping / sanity clamping
        if 'estimated_cost_inr_crore' in X_out.columns:
            X_out['estimated_cost_inr_crore'] = pd.to_numeric(X_out['estimated_cost_inr_crore'], errors='coerce').clip(upper=1e8).fillna(0.0)
        if 'title_dispute_rate_percent' in X_out.columns:
            X_out['title_dispute_rate_percent'] = pd.to_numeric(X_out['title_dispute_rate_percent'], errors='coerce').clip(lower=0.0, upper=100.0).fillna(0.0)

        # 3. Interactions
        if 'state' in X_out.columns and 'project_type' in X_out.columns:
            X_out['state_project_type'] = X_out['state'].astype(str) + "_" + X_out['project_type'].astype(str)
            
        return X_out


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies np.log1p to variables with heavy right-skew before scaling.
    """
    def __init__(self, cols=None):
        self.cols = cols if cols else ['project_cost_cr', 'land_area_hectares', 'affected_families_count']
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X_out = X.copy()
        for col in self.cols:
            if col in X_out.columns:
                # Convert to numeric just in case, coerce errors to NaN, fill with 0
                X_out[col] = pd.to_numeric(X_out[col], errors='coerce').fillna(0)
                # Apply log1p safely
                X_out[col] = np.log1p(np.maximum(0, X_out[col]))
        return X_out

class OOFTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed Target Encoder (m-estimate) with Out-Of-Fold regularization
    to prevent target leakage during training.
    """
    def __init__(self, cols=None, m=10.0, cv=5, random_state=42):
        self.cols = cols
        self.m = m
        self.cv = cv
        self.random_state = random_state
        
    def fit(self, X, y):
        y = np.array(y)
        self.global_mean_ = np.mean(y)
        self.cols_to_encode_ = self.cols if self.cols is not None else list(X.select_dtypes(include=['object', 'category']).columns)
        self.mapping_ = {}
        
        # Precompute global mappings for transform() on test data
        for col in self.cols_to_encode_:
            if col in X.columns:
                df_temp = pd.DataFrame({col: X[col], 'target': y})
                agg = df_temp.groupby(col)['target'].agg(['sum', 'count'])
                smoothed = (agg['sum'] + self.m * self.global_mean_) / (agg['count'] + self.m)
                self.mapping_[col] = smoothed.to_dict()
                
        return self
        
    def fit_transform(self, X, y=None):
        if y is None:
            return self.fit(X).transform(X)
            
        y = np.array(y)
        self.fit(X, y)
        X_out = X.copy()
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        for col in self.cols_to_encode_:
            if col not in X.columns:
                continue
                
            out_col = np.zeros(len(X))
            
            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr = y[train_idx]
                
                df_temp = pd.DataFrame({col: X_tr[col], 'target': y_tr})
                agg = df_temp.groupby(col)['target'].agg(['sum', 'count'])
                smoothed = (agg['sum'] + self.m * self.global_mean_) / (agg['count'] + self.m)
                
                out_col[val_idx] = X_val[col].map(smoothed).fillna(self.global_mean_)
                
            # Replace the categorical column with the encoded float
            X_out[col] = out_col.astype(float)
            
        return X_out

    def transform(self, X, y=None):
        X_out = X.copy()
        for col in self.cols_to_encode_:
            if col in X_out.columns:
                # Use global mapping learned during fit
                X_out[col] = X_out[col].map(self.mapping_.get(col, {})).fillna(self.global_mean_).astype(float)
        # Ensure all columns are numeric floats and no residual NaNs or strings
        bool_map = {
            True: 1.0, False: 0.0, 1: 1.0, 0: 0.0, 1.0: 1.0, 0.0: 0.0,
            'true': 1.0, 'false': 0.0, 'True': 1.0, 'False': 0.0, 'TRUE': 1.0, 'FALSE': 0.0,
            't': 1.0, 'f': 0.0, 'T': 1.0, 'F': 0.0,
            'yes': 1.0, 'no': 0.0, 'Yes': 1.0, 'No': 0.0, 'YES': 1.0, 'NO': 0.0,
            'y': 1.0, 'n': 0.0, 'Y': 1.0, 'N': 0.0,
            '1': 1.0, '0': 0.0
        }
        for col in X_out.columns:
            if col not in self.cols_to_encode_:
                if pd.api.types.is_bool_dtype(X_out[col]) or col == 'local_protest_flag':
                    if not pd.api.types.is_numeric_dtype(X_out[col]) or pd.api.types.is_bool_dtype(X_out[col]):
                        mapped = X_out[col].map(bool_map)
                        X_out[col] = pd.to_numeric(mapped, errors='coerce').fillna(0.0).astype(float)
                    else:
                        X_out[col] = pd.to_numeric(X_out[col], errors='coerce').fillna(0.0).astype(float)
                else:
                    X_out[col] = pd.to_numeric(X_out[col], errors='coerce').fillna(0.0).astype(float)
        return X_out

class SMOTENCDynamicWrapper(BaseEstimator):
    """
    Wrapper for SMOTE-NC that dynamically fetches categorical indices.
    If no categorical features exist, it falls back to regular SMOTE.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.sampler_ = None
        
    def fit_resample(self, X, y):
        cat_indices = []
        if isinstance(X, pd.DataFrame):
            cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
            cat_indices = [X.columns.get_loc(c) for c in cat_cols]
            
        # Avoid unnecessary warnings or errors if data is not imbalanced
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) <= 1 or min(counts) == max(counts):
            return X, y
            
        min_samples = min(counts)
        k_neighbors = min(5, min_samples - 1)
        if k_neighbors < 1:
            return X, y

        if len(cat_indices) > 0:
            self.sampler_ = SMOTENC(categorical_features=cat_indices, k_neighbors=k_neighbors, random_state=self.random_state)
        else:
            from imblearn.over_sampling import SMOTE
            self.sampler_ = SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)
            
        X_res, y_res = self.sampler_.fit_resample(X, y)
        return X_res, y_res
        
    def fit(self, X, y=None):
        self.fit_resample(X, y)
        return self

    def transform(self, X):
        return X

def get_preprocessing_pipeline(cat_cols=None, log_cols=None, te_cols=None, use_smote=False):
    """
    Constructs the leakage-free preprocessing pipeline.
    """
    steps = [
        ('feature_engineer', FeatureEngineer()),
        ('tracker', DynamicFeatureTracker(cat_cols=cat_cols)),
        ('log_transform', LogTransformer(cols=log_cols)),
    ]
    
    if use_smote:
        steps.append(('smote_nc', SMOTENCDynamicWrapper(random_state=42)))
        
    steps.append(('target_encoder', OOFTargetEncoder(cols=te_cols, m=10.0, cv=5)))
    
    return ImblearnPipeline(steps)
