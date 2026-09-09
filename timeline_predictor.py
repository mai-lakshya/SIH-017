import numpy as np
import pandas as pd
import torch
torch.set_num_threads(4)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from sksurv.util import Surv
import joblib

def create_structured_survival_array(status, duration_days):
    """
    Format target as structured array for sksurv.
    """
    return Surv.from_arrays(event=status.astype(bool), time=duration_days.astype(float))

class DeepSurvMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=2, dropout_p=0.2):
        super(DeepSurvMLP, self).__init__()
        layers = []
        current_dim = input_dim
        
        # We start with e.g. 128, then halve it
        next_dim = 128
        
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.BatchNorm1d(next_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout_p))
            current_dim = next_dim
            next_dim = max(32, next_dim // 2)
            
        layers.append(nn.Linear(current_dim, 1)) # Outputs log hazard ratio
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def cox_ph_loss(log_h, events, durations):
    """
    Compute Cox Negative Log-Likelihood.
    """
    sorted_indices = torch.argsort(durations, descending=True)
    log_h_sorted = log_h[sorted_indices]
    events_sorted = events[sorted_indices]
    
    risk_scores = torch.exp(log_h_sorted)
    cumsum_risk = torch.cumsum(risk_scores, dim=0)
    log_risk = torch.log(cumsum_risk + 1e-15)
    
    uncensored_likelihood = log_h_sorted - log_risk
    loss = -torch.sum(uncensored_likelihood * events_sorted) / (torch.sum(events_sorted) + 1e-15)
    return loss

class DeepSurvModel:
    def __init__(self, epochs=100, batch_size=64, lr=1e-3, patience=10, hidden_layers=2, dropout_p=0.2):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.hidden_layers = hidden_layers
        self.dropout_p = dropout_p
        self.model = None
        self.device = torch.device('cpu')
        
    def fit(self, X, status, duration):
        X_np = X.values if hasattr(X, 'values') else X
        X_np = X_np.astype(np.float32)
        status_np = np.array(status).astype(np.float32)
        duration_np = np.array(duration).astype(np.float32)
        
        self.model = DeepSurvMLP(input_dim=X_np.shape[1], hidden_layers=self.hidden_layers, dropout_p=self.dropout_p).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        dataset = TensorDataset(torch.tensor(X_np), torch.tensor(status_np), torch.tensor(duration_np))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_x, batch_event, batch_dur in dataloader:
                batch_x = batch_x.to(self.device)
                batch_event = batch_event.to(self.device)
                batch_dur = batch_dur.to(self.device)
                
                optimizer.zero_grad()
                log_h = self.model(batch_x)
                loss = cox_ph_loss(log_h, batch_event, batch_dur)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step()
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                break
                
        return self
        
    def predict(self, X):
        self.model.eval()
        X_np = X.values if hasattr(X, 'values') else X
        X_np = X_np.astype(np.float32)
        with torch.no_grad():
            log_h = self.model(torch.tensor(X_np).to(self.device))
        return log_h.cpu().numpy().flatten()


class NonLinearTimelinePredictor:
    def __init__(self, random_state=42, rsf_params=None, deepsurv_params=None):
        self.random_state = random_state
        
        # Configure RSF (compact ensemble for standalone portability <50MB)
        default_rsf = {'n_estimators': 60, 'min_samples_split': 10, 'max_features': 'sqrt', 'n_jobs': -1}
        if rsf_params:
            default_rsf.update(rsf_params)
        self.rsf = RandomSurvivalForest(random_state=random_state, **default_rsf)
        
        # Configure DeepSurv
        default_ds = {'epochs': 50, 'batch_size': 128, 'lr': 1e-3, 'hidden_layers': 2, 'dropout_p': 0.2, 'patience': 10}
        if deepsurv_params:
            default_ds.update(deepsurv_params)
        self.deepsurv = DeepSurvModel(**default_ds)
        
        self.max_observed_time_ = 3650.0

    def fit(self, X, status, duration):
        y_surv = create_structured_survival_array(status, duration)
        
        self.rsf.fit(X, y_surv)
        self.deepsurv.fit(X, status, duration)
        
        self.max_observed_time_ = np.max(duration)
        return self
        
    def _prepare_rsf_features(self, X):
        from survival_features import prepare_survival_features
        if hasattr(self, 'rsf') and self.rsf is not None:
            n_in = getattr(self.rsf, 'n_features_in_', None)
            expected = getattr(self.rsf, 'feature_names_in_', None)
            if expected is None and n_in is not None and hasattr(X, 'shape') and X.shape[1] == n_in:
                return X
        expected = getattr(self.rsf, 'feature_names_in_', None)
        return prepare_survival_features(X, expected)

    def _prepare_deepsurv_features(self, X):
        from survival_features import BASE_PIPELINE_FEATURES
        if hasattr(self, 'deepsurv') and self.deepsurv is not None and hasattr(self.deepsurv, 'model') and self.deepsurv.model is not None:
            expected_dim = self.deepsurv.model.net[0].in_features
            if isinstance(X, pd.DataFrame):
                if X.shape[1] == expected_dim:
                    return X
                if expected_dim == len(BASE_PIPELINE_FEATURES):
                    cols = [c for c in BASE_PIPELINE_FEATURES if c in X.columns]
                    if len(cols) == expected_dim:
                        return X[cols]
            if hasattr(X, 'shape') and X.shape[1] != expected_dim:
                if X.shape[1] > expected_dim:
                    return X.iloc[:, :expected_dim] if isinstance(X, pd.DataFrame) else X[:, :expected_dim]
                elif X.shape[1] < expected_dim:
                    pad_width = expected_dim - X.shape[1]
                    if isinstance(X, pd.DataFrame):
                        pad_df = pd.DataFrame(np.zeros((len(X), pad_width)), index=X.index, columns=[f"_pad_{i}" for i in range(pad_width)])
                        return pd.concat([X, pad_df], axis=1)
                    else:
                        return np.pad(X, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        return X

    def predict(self, X):
        X_rsf = self._prepare_rsf_features(X)
        return self.rsf.predict(X_rsf)

    def predict_survival_function(self, X):
        X_rsf = self._prepare_rsf_features(X)
        return self.rsf.predict_survival_function(X_rsf)

    def predict_cumulative_hazard_function(self, X):
        X_rsf = self._prepare_rsf_features(X)
        return self.rsf.predict_cumulative_hazard_function(X_rsf)

    def get_dynamic_risk_threshold(self, X):
        """
        Determines median survival time from RSF, adjusted by DeepSurv risk score.
        """
        X_rsf = self._prepare_rsf_features(X)
        X_ds = self._prepare_deepsurv_features(X)
        surv_funcs = self.rsf.predict_survival_function(X_rsf)
        ds_risk_scores = self.deepsurv.predict(X_ds)
        
        # Normalize deepsurv risk scores to act as a multiplier (basic scaling)
        ds_multiplier = np.exp(ds_risk_scores - np.mean(ds_risk_scores))
        
        median_survival_times = []
        for i, fn in enumerate(surv_funcs):
            times = fn.x
            probs = fn.y
            
            idx = np.where(probs <= 0.5)[0]
            if len(idx) > 0:
                base_time = times[idx[0]]
            else:
                base_time = self.max_observed_time_
                
            # Combine logic: higher DeepSurv risk means shorter time (so we divide by multiplier)
            adjusted_time = base_time / max(0.1, ds_multiplier[i])
            median_survival_times.append(adjusted_time)
                
        return np.array(median_survival_times)

    def predict_time_to_delay(self, X):
        return self.get_dynamic_risk_threshold(X)

    def evaluate(self, X_train, y_train_status, y_train_duration, X_test, y_test_status, y_test_duration):
        y_train_surv = create_structured_survival_array(y_train_status, y_train_duration)
        y_test_surv = create_structured_survival_array(y_test_status, y_test_duration)
        
        risk_scores = self.rsf.predict(X_test)
        
        c_index_uno, _, _, _, _ = concordance_index_ipcw(y_train_surv, y_test_surv, risk_scores)
        
        horizons = np.array([90, 180, 270, 365])
        max_time = y_test_surv['time'].max()
        min_time = y_test_surv['time'].min()
        valid_horizons = horizons[(horizons >= min_time) & (horizons <= max_time - 1e-5)]
        
        if len(valid_horizons) > 0:
            surv_funcs = self.rsf.predict_survival_function(X_test)
            surv_probs = np.vstack([fn(valid_horizons) for fn in surv_funcs])
            ibs = integrated_brier_score(y_train_surv, y_test_surv, surv_probs, valid_horizons)
        else:
            ibs = np.nan
            
        return {
            'c_index_uno': c_index_uno,
            'integrated_brier_score': ibs
        }

    def save(self, filepath):
        filepath = str(filepath)
        joblib.dump(self, filepath, compress=3)
        if hasattr(self, 'rsf') and self.rsf is not None:
            rsf_path = filepath.replace("timeline.joblib", "rsf_only.joblib")
            if rsf_path != filepath:
                try:
                    joblib.dump(self.rsf, rsf_path, compress=3)
                except Exception:
                    pass

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
