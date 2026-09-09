# NEXUS-XAI: National Executive XAI Utility System

> **Land Acquisition Delay Prediction, Survival Modeling & Statutory Risk Governance for Indian Mega-Infrastructure**  
> *Developed for the Ministry of Rural Development · Smart India Hackathon (SIH)*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Survey of India Compliant](https://img.shields.io/badge/GIS-Survey_of_India_Compliant-green.svg)](dashboard/india_states.geojson)
[![Hardware Acceleration](https://img.shields.io/badge/Hardware-DirectML%20%7C%20NPU%20%7C%20ONNX-blueviolet.svg)](continuous_learning.py)

---

## Complete Master Documentation
For the complete step-by-step architectural breakdown, mathematical formulations, statutory legal context (RFCTLARR Act 2013), and operational guidelines, please see:
**[`DOCUMENTATION.md`](DOCUMENTATION.md)**

---

## Key Highlights

1. **Statutory RFCTLARR Act 2013 Compliance Engine**:
   - Explicit modeling of Section 11 preliminary notifications and the **Section 19(7) 365-day statutory lapse horizon**.
   - Solatium (100%), Rural Multiplier ($1.0\times$ to $2.0\times$), and R&R entitlements under Schedule II.

2. **Dual-Paradigm Machine Learning Architecture**:
   - **Paradigm 1 (Stacking Ensemble)**: XGBoost + LightGBM + CatBoost + ExtraTrees with a Logistic Regression Meta-Learner and Platt Scaling ($ECE \le 0.015$, $AUC \ge 0.88$).
   - **Paradigm 2 (Survival Analysis)**: Random Survival Forests (RSF) + DeepSurv evaluating non-linear survival curves $S(t)$ at 90d, 180d, 270d, 365d, 540d, and 730d ($C\text{-Index} \ge 0.9028$).
   - **Conformal Prediction**: Non-parametric 90% confidence intervals for delay day bounds.

3. **Explainable AI (XAI) & Prescriptive Mitigation Engine**:
   - **TreeSHAP Attributions**: Exact Shapley decomposition explaining why delays occur.
   - **Statutory Prescriptions**: Automated actionable recommendations mapped to RFCTLARR statutory interventions with expected delay reductions (days saved) and financial ROI.

4. **Continuous Learning Pipeline & Autonomous MLOps**:
   - **Daily Drift Surveillance**: Population Stability Index (PSI) computed across all features daily at midnight (`00:00`). $\text{PSI} > 0.20$ triggers automated retraining.
   - **Triple Validation Gate**: Candidate models must beat $C\text{-Index} \ge 0.88$, $ECE \le 0.10$, and $ROC\text{-}AUC \ge 0.85$ before deployment.
   - **Automated Versioning**: Checkpoints saved in `models/v*` with `model_card.json`, retaining the 3 latest versions.
   - **Hardware Acceleration**: Automatic binding to NPU / GPU via DirectML and ONNX Runtime.

5. **Database Persistence with Instant Retraining**:
   - Integrated SQLite database (`saved_analyses.db`).
   - Saving a project immediately appends it to the continuous training store (`indian_infrastructure_projects_dataset.csv`) and kicks off a non-blocking background retraining daemon.

6. **Survey of India GIS Cartography**:
   - High-performance Leaflet command center rendering 13,700+ infrastructure projects, state risk choropleths, and inter-state corridors using official Survey of India boundary GeoJSONs.

---

## Quickstart & Local Hosting

### 1. Installation

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install CPU-only PyTorch first (avoids heavy CUDA binaries on non-GPU setups)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install all dependencies
pip install -r requirements.txt
```

### 2. Launch Local Servers

#### A. FastAPI Web Application & API (Recommended)
```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
* **Landing Page**: [http://localhost:8000/](http://localhost:8000/)
* **Interactive Command Center**: [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
* **API Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc Technical Docs**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

#### B. Streamlit Analytics Dashboard
```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard.py --server.port 8501
```
* **Streamlit UI**: [http://localhost:8501/](http://localhost:8501/)

---

## Comprehensive Verification

To run the complete 9-pillar continuous learning test suite (Ingestion, PSI Drift, Retraining, Validation Gate, Model Versioning, NPUParity, APScheduler, Dashboard API, and Full End-to-End lifecycle):

```bash
python run_all_continuous_learning_tests.py
```

To run the unit test suite:
```bash
pytest -v
```

---

## Repository Structure

```text
SIh/
├── api.py                                  # FastAPI backend serving REST endpoints & SQLite persistence
├── continuous_learning.py                  # MLOps engine: Ingestion, PSI drift, Retraining, Gates, NPU
├── scheduler.py                            # APScheduler daemon for daily drift & weekly retraining jobs
├── hybrid_model.py                         # Stacking Ensemble & Calibrated Regressor (XGB, LGBM, Cat, ET)
├── timeline_predictor.py                   # Timeline Survival Analysis Engine (RSF + DeepSurv)
├── dual_paradigm_explainer.py              # TreeSHAP localized attribution engine
├── recommendation_engine.py                # Prescriptive AI mitigation and ROI calculation engine
├── dashboard.py                            # Streamlit exploratory analytics application
├── run_all_continuous_learning_tests.py    # Master 9-pillar continuous learning test runner
├── dashboard/                              # Apple-Design SPA (HTML/CSS/JS, Survey of India GeoJSON)
├── saved_analyses.db                       # Persistent SQLite database for saved analyses
├── indian_infrastructure_projects_dataset.csv # Continuous learning dataset (13,733+ records)
├── DOCUMENTATION.md                        # Master Architecture & Operational Documentation
└── requirements.txt                        # Pinned dependencies
```

---

## Statutory & Compliance Disclaimer
All geospatial boundary representations in this repository adhere to the **Survey of India** official cartographic standards. All legal risk classifications conform to the **Right to Fair Compensation and Transparency in Land Acquisition, Rehabilitation and Resettlement (RFCTLARR) Act, 2013**.
