from fastapi import FastAPI, HTTPException, Security, Request, Depends, Query
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import os
import re
import logging
import json
import math
import random
import time
import datetime
import threading
import jwt
import sqlite3
import uuid
from typing import Optional, List, Dict, Any
from starlette.concurrency import run_in_threadpool

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    has_slowapi = True
except ImportError:
    has_slowapi = False
    class RateLimitExceeded(Exception):
        pass
    def _rate_limit_exceeded_handler(request, exc):
        pass
    class Limiter:
        def __init__(self, *args, **kwargs):
            pass
        def limit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    def get_remote_address(request=None):
        return "127.0.0.1"

from risk_analysis_system import RiskAnalysisSystem
from monitor import ModelMonitor
from recommendation_engine import calculate_roi_for_recommendation
from ai_advisor import AIAdvisor, PromptSecurityValidator, DomainGroundingValidator, IndianContextNormalizer
from remoteness.remoteness_score import evaluate_remoteness

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Phase 9: Security & Rate Limiting ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Land Acquisition Risk API", version="2.0")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
if has_slowapi:
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Authentication Configuration ---
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-in-production")
ALGORITHM = "HS256"
API_KEY = os.getenv("API_KEY", "super-secret-token")
DEMO_EMAIL = os.getenv("DEMO_EMAIL", "demo@ministry.gov")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "SIH2024Demo")
VALID_ROLES = {"Ministry Official", "State Administrator", "District Officer", "Project Implementer"}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
http_bearer = HTTPBearer(auto_error=False)

async def get_current_user(
    request: Request,
    api_key_val: Optional[str] = Security(api_key_header),
    auth_header: Optional[HTTPAuthorizationCredentials] = Security(http_bearer)
) -> Dict[str, Any]:
    """
    Validates either an X-API-Key header OR an Authorization: Bearer <jwt> header.
    Rejects with 401/403 if neither is present or valid (no silent fallback).
    """
    # 1. Check API Key header
    if api_key_val is not None:
        if api_key_val == API_KEY:
            return {"email": "api-key@ministry.gov", "role": "Ministry Official", "auth_type": "api_key"}
        raise HTTPException(status_code=403, detail="Invalid API Key")

    # 2. Check Bearer JWT token header
    token = None
    if auth_header and auth_header.scheme.lower() == "bearer":
        token = auth_header.credentials
    elif "authorization" in request.headers:
        raw_auth = request.headers["authorization"].strip()
        if raw_auth.lower().startswith("bearer "):
            token = raw_auth[7:].strip()

    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Could not validate token")

    # 3. Reject if neither is present
    raise HTTPException(status_code=401, detail="Authentication required: Provide X-API-Key or Bearer token")

# Backward compatibility alias
get_api_key = get_current_user

class LoginRequest(BaseModel):
    email: str
    password: str
    role: str

@app.post("/auth/login")
@limiter.limit("10/minute")
async def login(request: Request, credentials: LoginRequest):
    """
    Demo user login. Generates a signed JWT access token valid for 24 hours.
    """
    if credentials.role not in VALID_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role '{credentials.role}'. Must be one of: {sorted(list(VALID_ROLES))}"
        )
    if credentials.email != DEMO_EMAIL or credentials.password != DEMO_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    now = datetime.datetime.now(datetime.timezone.utc)
    exp = now + datetime.timedelta(hours=24)
    payload = {
        "email": credentials.email,
        "role": credentials.role,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp())
    }
    access_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": credentials.role,
        "email": credentials.email
    }

# --- Geographic Reference Data (28 States + 8 Union Territories) ---
INDIA_STATE_CENTROIDS: Dict[str, tuple[float, float]] = {
    # 28 States
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
    # 8 Union Territories
    "Andaman and Nicobar Islands": (11.7401, 92.6586),
    "Chandigarh": (30.7333, 76.7794),
    "Dadra and Nagar Haveli and Daman and Diu": (20.1809, 73.0169),
    "Delhi": (28.7041, 77.1025),
    "Jammu and Kashmir": (33.7782, 76.5762),
    "Ladakh": (34.1526, 77.5771),
    "Lakshadweep": (10.5667, 72.6417),
    "Puducherry": (11.9416, 79.8083),
}

STATE_ABBREVIATIONS: Dict[str, str] = {
    "Andhra Pradesh": "AP", "Arunachal Pradesh": "AR", "Assam": "AS", "Bihar": "BR",
    "Chhattisgarh": "CG", "Goa": "GA", "Gujarat": "GJ", "Haryana": "HR",
    "Himachal Pradesh": "HP", "Jharkhand": "JH", "Karnataka": "KA", "Kerala": "KL",
    "Madhya Pradesh": "MP", "Maharashtra": "MH", "Manipur": "MN", "Meghalaya": "ML",
    "Mizoram": "MZ", "Nagaland": "NL", "Odisha": "OD", "Punjab": "PB",
    "Rajasthan": "RJ", "Sikkim": "SK", "Tamil Nadu": "TN", "Telangana": "TS",
    "Tripura": "TR", "Uttar Pradesh": "UP", "Uttarakhand": "UK", "West Bengal": "WB",
    "Andaman and Nicobar Islands": "AN", "Chandigarh": "CH",
    "Dadra and Nagar Haveli and Daman and Diu": "DN", "Delhi": "DL",
    "Jammu and Kashmir": "JK", "Ladakh": "LA", "Lakshadweep": "LD", "Puducherry": "PY"
}

def derive_project_status(row: Dict[str, Any]) -> str:
    """
    Maps project phase based on fund_disbursement_percent and sia_approval_status:
    - fund >= 75% = Possessed
    - fund >= 25% = Compensating
    - SIA approved/exempted = Awarded
    - SIA pending or section_11_notification_days > 0 = Notified
    - else Proposed
    """
    for col in ['status', 'project_status', 'phase', 'acquisition_status']:
        val = str(row.get(col, '')).strip().capitalize()
        if val in ["Proposed", "Notified", "Awarded", "Compensating", "Possessed"]:
            return val

    try:
        fund_pct = float(row.get('fund_disbursement_percent', 0.0) or 0.0)
    except (ValueError, TypeError):
        fund_pct = 0.0

    sia_raw = str(row.get('sia_approval_status', '')).strip().lower()

    try:
        sec11_days = float(row.get('section_11_notification_days', 0) or 0)
    except (ValueError, TypeError):
        sec11_days = 0.0

    if fund_pct >= 75.0:
        return "Possessed"
    elif fund_pct >= 25.0:
        return "Compensating"
    elif sia_raw in ['approved', 'exempted']:
        return "Awarded"
    elif sia_raw == 'pending' or sec11_days > 0:
        return "Notified"
    else:
        return "Proposed"

_DISTRICT_COORDS: Dict[str, list] = {}

def get_district_coordinates(state: str, district: str) -> Optional[tuple[float, float]]:
    global _DISTRICT_COORDS
    if not _DISTRICT_COORDS:
        for p in ['district_coordinates.json', 'scratch/district_coordinates.json']:
            if os.path.exists(p):
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        _DISTRICT_COORDS = json.load(f)
                    break
                except Exception:
                    pass

    key = f"{state}|{district}"
    if key in _DISTRICT_COORDS:
        c = _DISTRICT_COORDS[key]
        return float(c[0]), float(c[1])

    s_low = state.strip().lower()
    d_low = district.strip().lower()
    for k, v in _DISTRICT_COORDS.items():
        if '|' in k:
            ks, kd = k.split('|', 1)
            if ks.lower() == s_low and (kd.lower() == d_low or d_low in kd.lower() or kd.lower() in d_low):
                return float(v[0]), float(v[1])
    return None

def derive_coordinates(row: Dict[str, Any], project_id: str, state: str, district: str = "", intra_index: int = 0) -> tuple[float, float]:
    """
    Extracts true coordinates if present in row.
    Otherwise, resolves exact geographic district coordinates.
    When multiple projects share the same district/location (intra_index > 0),
    applies a compact deterministic golden-spiral dispersion so markers do not stack on the exact same pixel.
    """
    for lat_col in ['latitude', 'lat', 'Latitude', 'LATITUDE']:
        for lon_col in ['longitude', 'lon', 'long', 'Longitude', 'LONGITUDE']:
            if lat_col in row and lon_col in row and row[lat_col] is not None and row[lon_col] is not None:
                try:
                    lat_v = float(row[lat_col])
                    lon_v = float(row[lon_col])
                    if not (math.isnan(lat_v) or math.isnan(lon_v)):
                        return round(lat_v, 4), round(lon_v, 4)
                except (ValueError, TypeError):
                    pass

    dist_coords = get_district_coordinates(state, district) if district else None
    if dist_coords:
        base_lat, base_lon = dist_coords
    else:
        base_coords = INDIA_STATE_CENTROIDS.get(state)
        if not base_coords:
            for s_name, s_coords in INDIA_STATE_CENTROIDS.items():
                if s_name.lower() == state.strip().lower():
                    base_coords = s_coords
                    break
        if not base_coords:
            base_coords = (20.5937, 78.9629)
        base_lat, base_lon = base_coords

    if intra_index == 0:
        return round(base_lat, 4), round(base_lon, 4)

    # Golden-ratio spiral dispersion around the district center (~300m to 2.5km)
    angle = intra_index * 2.3999632
    radius = min(0.028, 0.0035 * math.sqrt(intra_index))
    lat_offset = radius * math.cos(angle)
    cos_lat = math.cos(math.radians(base_lat)) if abs(base_lat) < 89 else 1.0
    lon_offset = (radius * math.sin(angle)) / (cos_lat if abs(cos_lat) > 0.1 else 1.0)

    return round(base_lat + lat_offset, 4), round(base_lon + lon_offset, 4)

def derive_project_name(row: Dict[str, Any], state: str, district: str, project_type: str, idx: int) -> str:
    """Extracts project name or synthesizes a clean descriptive title from project attributes."""
    for col in ['project_name', 'name', 'title', 'Project_Name']:
        val = str(row.get(col, '')).strip()
        if val and val.lower() not in ['nan', 'none', '']:
            return val

    clean_type = str(project_type).replace('_', ' ').title()
    if district and district not in ['Unknown', 'nan', '']:
        return f"{state} {clean_type} Corridor ({district})"
    return f"{state} {clean_type} Expansion Phase {idx % 5 + 1}"

# --- Geo Memory Cache (5-Minute TTL) ---
_GEO_CACHE: Dict[str, Any] = {
    "data": None,
    "timestamp": 0.0,
    "details_by_id": {},
    "raw_rows_by_id": {}
}
_GEO_CACHE_LOCK = threading.Lock()
GEO_CACHE_TTL_SECONDS = 300  # 5 minutes

class ProjectPayload(BaseModel):
    project_id: Optional[str] = 'NHAI-UNKNOWN'
    state: str
    district: Optional[str] = 'Unknown'
    land_area_hectares: float = Field(..., gt=0.0, description="Land area in hectares, must be positive")
    land_area_log: Optional[float] = 5.0
    project_type: str
    terrain_type: str
    estimated_cost_inr_crore: float = Field(..., gt=0.0, description="Estimated project cost in INR Crores, must be positive")
    affected_families_count: Optional[int] = Field(default=500, ge=0)
    title_dispute_rate_percent: Optional[float] = Field(default=5.0, ge=0.0, le=100.0)
    local_protest_flag: Optional[bool] = False
    compensation_multiplier_demand: Optional[float] = Field(default=1.5, ge=0.0)
    sia_approval_status: Optional[str] = 'Pending'
    sia_approval_status_risk_score: Optional[float] = 0.5
    section_11_notification_days: Optional[int] = 30
    forest_clearance_status: Optional[str] = 'Not_Required'
    forest_clearance_status_risk_score: Optional[float] = 0.5
    fund_disbursement_percent: Optional[float] = Field(default=10.0, ge=0.0, le=100.0)
    project_start_year: Optional[int] = 2022
    project_age_years: Optional[int] = 1
    schedule_tasks: Optional[List[Dict[str, Any]]] = None
    target_completion_days: Optional[float] = None
    latitude: Optional[float] = Field(default=None, description="Project site latitude in decimal degrees")
    longitude: Optional[float] = Field(default=None, description="Project site longitude in decimal degrees")
    road_type: Optional[str] = Field(default=None, description="Road connectivity type near site")
    road_connectivity_type: Optional[str] = Field(default=None, description="Alias for road_type")
    address: Optional[str] = Field(default=None, description="Address or village name for geocoding")

class RemotenessRequest(BaseModel):
    latitude: Optional[float] = Field(default=None, description="Site latitude")
    longitude: Optional[float] = Field(default=None, description="Site longitude")
    address: Optional[str] = Field(default=None, description="Address, village, or town name to geocode")
    project_type: Optional[str] = Field(default=None, description="Infrastructure sector (e.g. Highway, Railway)")
    district: Optional[str] = Field(default=None, description="District name")
    state: Optional[str] = Field(default=None, description="State or UT name")
    road_type: Optional[str] = Field(default=None, description="Road classification")
    terrain_type: Optional[str] = Field(default=None, description="Terrain classification (plain, hilly, coastal, forest_tribal)")
    allow_online: Optional[bool] = Field(default=False, description="Enable live OSM Overpass/Nominatim queries")

class AIAdvisoryRequest(BaseModel):
    query: str
    context: Optional[str] = None
    project_metadata: Optional[Dict[str, Any]] = None

class SimulationPayload(BaseModel):
    baseline: ProjectPayload
    interventions: Dict[str, Any]

class SaveAnalysisRequest(BaseModel):
    project_name: str
    state: Optional[str] = None
    district: Optional[str] = None
    input_payload: Dict[str, Any]

# --- Phase 10: Persistent Memory Storage (saved_analyses.db) ---
SAVED_ANALYSES_DB = os.getenv("SAVED_ANALYSES_DB", "saved_analyses.db")
_SAVED_DB_LOCK = threading.RLock()

def init_saved_analyses_db():
    with _SAVED_DB_LOCK:
        conn = sqlite3.connect(SAVED_ANALYSES_DB, timeout=30.0)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS saved_analyses (
                        id TEXT PRIMARY KEY,
                        project_name TEXT NOT NULL,
                        created_by_email TEXT,
                        state TEXT,
                        district TEXT,
                        project_type TEXT,
                        latitude REAL,
                        longitude REAL,
                        input_payload TEXT,
                        delay_probability REAL,
                        risk_tier TEXT,
                        predicted_delay_days INTEGER,
                        composite_risk_score REAL,
                        created_at TEXT
                    )
                """)
            logging.info("Initialized persistent saved_analyses table in %s", SAVED_ANALYSES_DB)
        finally:
            conn.close()

def get_saved_db_connection():
    conn = sqlite3.connect(SAVED_ANALYSES_DB, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize DB on module load as well
init_saved_analyses_db()

# Global variables
system: RiskAnalysisSystem = None
monitor: ModelMonitor = None

@app.on_event("startup")
def load_artifacts():
    global system, monitor
    init_saved_analyses_db()
    try:
        pipeline_path = 'pipeline.joblib'
        ensemble_path = 'ensemble.joblib'
        timeline_path = 'models_new/timeline.joblib' if os.path.exists('models_new/timeline.joblib') and os.path.getsize('models_new/timeline.joblib') > 1000 else 'timeline.joblib'

        # Check if files are missing or are 130-byte LFS pointer text files
        needs_retrain = False
        for p in [pipeline_path, ensemble_path, timeline_path]:
            if not os.path.exists(p) or os.path.getsize(p) < 1000:
                logging.warning(f"Artifact {p} is missing or an unresolved LFS pointer! Triggering auto-training...")
                needs_retrain = True
                break

        if needs_retrain:
            import retrain_all
            retrain_all.main()

        system = RiskAnalysisSystem(
            pipeline_path=pipeline_path,
            ensemble_path=ensemble_path,
            timeline_path=timeline_path
        )
        monitor = ModelMonitor()
        logging.info("[READY] RiskAnalysisSystem and Monitor successfully loaded and ready.")

        # Initialize Continuous Learning Automated Scheduler
        try:
            from scheduler import start_scheduler
            start_scheduler()
            logging.info("[STARTED] Continuous Learning Scheduler (APScheduler) started.")
        except Exception as se:
            logging.warning(f"Could not start Continuous Learning Scheduler: {se}")
    except Exception as e:
        logging.error(f"Failed to load artifacts: {e}", exc_info=True)

@app.get("/health")
@app.head("/health")
def health_check():
    return {
        "status": "healthy",
        "system_ready": system is not None,
        "monitor_ready": monitor is not None,
        "meta_coefficients": getattr(system.explainer, 'meta_coefficients', {}) if (system and system.explainer) else {}
    }

@app.get("/")
@app.head("/")
@app.get("/home")
@app.head("/home")
@app.get("/landing")
@app.head("/landing")
def serve_landing():
    """Task 1: Standalone Landing Page"""
    path = "dashboard/landing.html"
    if os.path.exists(path):
        return FileResponse(path)
    return RedirectResponse(url="/docs")

@app.get("/dashboard")
@app.head("/dashboard")
@app.get("/model-governance")
@app.get("/prescriptive-ai")
def serve_dashboard():
    """Task 3: Standalone Dashboard Workbench"""
    path = "dashboard/index.html"
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Dashboard not found")

@app.get("/map")
def serve_map():
    path = "dashboard/screens/location_analysis_map.html"
    if os.path.exists(path):
        return FileResponse(path)
    return FileResponse("dashboard/index.html")

@app.get("/methodology")
def serve_methodology():
    """Task 2: Standalone Technical Methodology Page"""
    path = "dashboard/methodology.html"
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Methodology not found")

@app.get("/india_states.geojson")
def serve_india_geojson():
    from fastapi.responses import FileResponse
    for path in ["dashboard/india_states.geojson", "dashboard/screens/india_states.geojson"]:
        if os.path.exists(path):
            return FileResponse(path, media_type="application/geo+json")
    raise HTTPException(status_code=404, detail="GeoJSON not found")

@app.get("/india_national_boundary.geojson")
def serve_india_national_boundary():
    from fastapi.responses import FileResponse
    for path in ["dashboard/india_national_boundary.geojson", "dashboard/screens/india_national_boundary.geojson"]:
        if os.path.exists(path):
            return FileResponse(path, media_type="application/geo+json")
    raise HTTPException(status_code=404, detail="National boundary GeoJSON not found")

@app.get("/jk_soi_patch.geojson")
def serve_jk_patch():
    from fastapi.responses import FileResponse
    for path in ["dashboard/jk_soi_patch.geojson", "dashboard/screens/jk_soi_patch.geojson"]:
        if os.path.exists(path):
            return FileResponse(path, media_type="application/geo+json")
    raise HTTPException(status_code=404, detail="Patch GeoJSON not found")

@app.get("/india_districts.geojson")
def serve_india_districts_geojson():
    from fastapi.responses import FileResponse
    for path in ["dashboard/india_districts.geojson", "dashboard/screens/india_districts.geojson"]:
        if os.path.exists(path):
            return FileResponse(path, media_type="application/geo+json")
    raise HTTPException(status_code=404, detail="Districts GeoJSON not found")




def _prepare_df(payload_dict: dict) -> pd.DataFrame:
    # Normalize and dynamically derive statutory clearance risk scores
    sia_raw = str(payload_dict.get('sia_approval_status', 'Pending')).strip()
    sia_norm = sia_raw.lower().replace(' ', '_').replace('-', '_')
    sia_score_map = {
        'approved': 0.0,
        'exempted': 0.0,
        'in_progress': 0.4,
        'pending': 0.75,
        'rejected': 1.0
    }
    payload_dict['sia_approval_status_risk_score'] = sia_score_map.get(sia_norm, 0.5)

    fc_raw = str(payload_dict.get('forest_clearance_status', 'Not_Required')).strip()
    fc_norm = fc_raw.lower().replace(' ', '_').replace('-', '_')
    fc_score_map = {
        'not_required': 0.0,
        'approved': 0.0,
        'stage_2': 0.2,
        'stage_1': 0.4,
        'stage_1_approved': 0.4,
        'in_progress': 0.6,
        'stage_1_pending': 0.8,
        'pending': 0.8,
        'rejected': 1.0
    }
    payload_dict['forest_clearance_status_risk_score'] = fc_score_map.get(fc_norm, 0.5)

    raw_payload = pd.DataFrame([payload_dict])
    for col in ['C_r', 'F_r', 'H_r', 'W_r', 'P_r']:
        if col not in raw_payload:
            raw_payload[col] = 0.5
    if 'section_11_notification_days' in raw_payload:
        raw_payload = raw_payload.drop(columns=['section_11_notification_days'])
    return raw_payload

def _extract_survival_curve(raw_payload: pd.DataFrame) -> List[Dict[str, Any]]:
    survival_curve = []
    try:
        surv_funcs = None
        if system.timeline_predictor and hasattr(system.timeline_predictor, 'predict_survival_function'):
            X_proc = system.pipeline.transform(raw_payload)
            surv_funcs = system.timeline_predictor.predict_survival_function(X_proc)
        elif system.timeline_predictor and hasattr(system.timeline_predictor, 'rsf') and system.timeline_predictor.rsf is not None:
            X_proc = system.pipeline.transform(raw_payload)
            surv_funcs = system.timeline_predictor.rsf.predict_survival_function(X_proc)
        
        if surv_funcs is not None and len(surv_funcs) > 0:
            fn = surv_funcs[0]
            sample_times = [0, 15, 30, 60, 90, 120, 150, 180, 240, 300, 365, 450, 500, 600, 730]
            max_t = float(fn.x[-1]) if len(fn.x) > 0 else 730.0
            for t in sample_times:
                if t <= max_t:
                    prob = float(fn(t))
                    survival_curve.append({"day": int(t), "survival_probability": round(prob, 4)})
    except Exception as e:
        logging.warning(f"Could not compute survival curve: {e}")
    return survival_curve

def calculate_prescriptive_actions(result: Dict[str, Any], project_cost_cr: float, raw_payload: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """Translates model recommendations into actionable prescriptive mitigations with dynamic ROI."""
    raw_recs = result.get('recommendations', [])
    project_cost_inr = project_cost_cr * 10_000_000
    delay_cost_per_day = max(100_000, (project_cost_inr * 0.12) / 365)

    try:
        X_sample = system.pipeline.transform(raw_payload) if (hasattr(system, 'pipeline') and system.pipeline is not None and raw_payload is not None) else None
    except Exception:
        X_sample = None
    model_inst = getattr(system, 'hybrid_predictor', None)

    prescriptive_actions = []
    seen_titles = set()
    template_cursor = {}

    for rec in raw_recs:
        try:
            roi_info = calculate_roi_for_recommendation(
                rec,
                project_cost=project_cost_inr,
                delay_cost_per_day=delay_cost_per_day,
                model=model_inst,
                X_sample=X_sample
            )
        except Exception as roi_err:
            logging.warning("ROI calculation failed for rec %s (%s); applying fallback", rec.get('issue'), roi_err)
            roi_info = {
                'estimated_delay_days_saved': 15.0,
                'cost_savings': delay_cost_per_day * 15.0,
                'roi_percentage': 150.0
            }

        t_key = rec.get('template_key') or rec.get('category') or rec.get('source', 'default')
        actions = rec.get('actions', [])

        title = None
        desc = None

        if actions:
            cursor = template_cursor.get(t_key, 0)
            while cursor < len(actions):
                candidate = actions[cursor].strip()
                if candidate.lower() not in seen_titles:
                    title = candidate
                    if cursor + 1 < len(actions):
                        desc = actions[cursor + 1].strip()
                        template_cursor[t_key] = cursor + 2
                    else:
                        desc = rec.get('expected_impact') or rec.get('issue', 'Operational mitigation intervention')
                        template_cursor[t_key] = cursor + 1
                    break
                cursor += 1
            if not title:
                continue
        else:
            candidate_issue = rec.get('issue', 'Mitigation Action').strip()
            if candidate_issue.lower() not in seen_titles:
                title = candidate_issue
                desc = rec.get('expected_impact', 'Operational intervention')

        if not title or title.lower() in seen_titles:
            continue

        seen_titles.add(title.lower())

        delay_saved = round(float(roi_info.get('estimated_delay_days_saved', 0.0)), 1)
        cost_savings_cr = round(float(roi_info.get('cost_savings', 0.0)) / 10_000_000, 2)
        roi_pct = round(float(roi_info.get('roi_percentage', 0.0)), 1)

        prescriptive_actions.append({
            "title": title,
            "description": desc,
            "issue": rec.get('issue', title),
            "actions": rec.get('actions', [title]),
            "priority": rec.get('priority', 'Medium'),
            "timeframe": rec.get('timeframe', 'Short-term'),
            "expected_impact": rec.get('expected_impact', 'Risk reduction'),
            "delay_saved_days": int(round(delay_saved)),
            "avoided_delay": delay_saved,
            "avoided_delay_days": delay_saved,
            "cost_saved_cr": cost_savings_cr,
            "cost_savings_cr": cost_savings_cr,
            "cost_savings": cost_savings_cr,
            "roi": roi_pct,
            "roi_percentage": roi_pct,
            "roi_percent": int(round(roi_pct)),
            "buffer_status": rec.get("buffer_status", "Active Schedule Path")
        })
    return prescriptive_actions

def normalize_terrain_type(val: Any) -> str:
    """
    Normalizes raw dataset terrain string to match Risk Predictor dropdown options:
    'Urban', 'Rural_Agri', 'Forest_Eco_Sensitive', 'Hilly', 'Tribal_Schedule_V'.
    """
    if not val or pd.isna(val):
        return "Rural_Agri"
    s = str(val).strip()
    clean = s.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    mapping = {
        "urban": "Urban",
        "rural_agri": "Rural_Agri",
        "rural": "Rural_Agri",
        "rural_agriculture": "Rural_Agri",
        "forest_eco_sensitive": "Forest_Eco_Sensitive",
        "forest": "Forest_Eco_Sensitive",
        "eco_sensitive": "Forest_Eco_Sensitive",
        "hilly": "Hilly",
        "hilly_difficult": "Hilly",
        "tribal_schedule_v": "Tribal_Schedule_V",
        "tribal": "Tribal_Schedule_V",
        "schedule_v": "Tribal_Schedule_V",
    }
    return mapping.get(clean, s)

def get_or_load_geo_cache(max_projects: Optional[int] = None, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Loads and computes geospatial project predictions across the entire dataset with high-speed vectorized processing and caching."""
    global _GEO_CACHE
    now = time.time()

    if not force_refresh and _GEO_CACHE["data"] is not None:
        if (now - _GEO_CACHE["timestamp"]) < GEO_CACHE_TTL_SECONDS:
            if max_projects is not None:
                return _GEO_CACHE["data"][:max_projects]
            return _GEO_CACHE["data"]

    with _GEO_CACHE_LOCK:
        now = time.time()
        if not force_refresh and _GEO_CACHE["data"] is not None:
            if (now - _GEO_CACHE["timestamp"]) < GEO_CACHE_TTL_SECONDS:
                if max_projects is not None:
                    return _GEO_CACHE["data"][:max_projects]
                return _GEO_CACHE["data"]

        csv_path = "indian_infrastructure_projects_dataset.csv"
        if not os.path.exists(csv_path):
            if os.path.exists("Revolution-main/indian_infrastructure_projects_dataset.csv"):
                csv_path = "Revolution-main/indian_infrastructure_projects_dataset.csv"
            else:
                logging.error("Projects dataset CSV not found at %s", csv_path)
                raise HTTPException(
                    status_code=500,
                    detail="Projects dataset not found: 'indian_infrastructure_projects_dataset.csv' is missing from the server."
                )

        start_time = time.perf_counter()
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.error("Failed to parse %s: %s", csv_path, e)
            raise HTTPException(status_code=500, detail=f"Failed to load projects dataset CSV: {e}")

        total_rows = len(df)
        logging.info("Vectorizing complete dataset (%d projects) for /projects/geo from %s...", total_rows, csv_path)

        records = df.to_dict('records')
        results = []
        details_by_id = {}
        raw_rows_by_id = {}
        district_counts: Dict[str, int] = {}

        for idx, raw_dict in enumerate(records):
            state = str(raw_dict.get('state', 'Unknown')).strip()
            district = str(raw_dict.get('district', 'Unknown')).strip()
            project_type = str(raw_dict.get('project_type', 'Infrastructure')).strip()

            proj_id = raw_dict.get('project_id')
            if not proj_id or pd.isna(proj_id) or str(proj_id).strip() in ['', 'nan']:
                state_abbr = STATE_ABBREVIATIONS.get(state, "IND")
                proj_id = f"{state_abbr}-{idx+1:05d}"
            else:
                proj_id = str(proj_id).strip()

            raw_dict['project_id'] = proj_id
            loc_key = f"{state}|{district}"
            intra_idx = district_counts.get(loc_key, 0)
            district_counts[loc_key] = intra_idx + 1

            proj_name = derive_project_name(raw_dict, state, district, project_type, idx)
            status = derive_project_status(raw_dict)
            lat, lon = derive_coordinates(raw_dict, proj_id, state, district, intra_idx)

            raw_crs = raw_dict.get('CRS')
            crs_val = float(raw_crs) if raw_crs is not None and not pd.isna(raw_crs) else 50.0
            crs_rounded = round(crs_val, 1)

            # Strictly calibrated on the Risk Legend:
            # Low: <= 25.0, Med: 26-50 (25.0 < CRS <= 50.0), High: > 50.0
            tier_val = "High" if crs_rounded > 50.0 else ("Medium" if crs_rounded > 25.0 else "Low")
            prob_val = 1.0 / (1.0 + math.exp(-0.06 * (crs_rounded - 48.0)))
            delay_days_val = max(0.0, crs_rounded * 2.8 - 30.0)
            med_surv = max(60, int(round(180 + (crs_rounded - 50) * 1.5)))

            delay_prob = round(prob_val * 100, 1)
            predicted_delay_days = int(round(delay_days_val))

            sec11_days = int(raw_dict.get('section_11_notification_days', 30) or 30)
            comp_mult = float(raw_dict.get('compensation_multiplier_demand', 1.5) or 1.5)
            aff_families = int(raw_dict.get('affected_families_count', 0) or 0)
            dispute_pct = float(raw_dict.get('title_dispute_rate_percent', 0.0) or 0.0)
            sia_status = str(raw_dict.get('sia_approval_status', 'Pending')).strip()
            fc_status = str(raw_dict.get('forest_clearance_status', 'Not_Required')).strip()
            raw_terrain = raw_dict.get('terrain_type') or raw_dict.get('Terrain_Type') or 'Rural_Agri'
            terrain = normalize_terrain_type(raw_terrain)
            cost_cr = float(raw_dict.get('estimated_cost_inr_crore', 0.0) or 0.0)
            land_ha = float(raw_dict.get('land_area_hectares', 0.0) or 0.0)
            fund_pct = float(raw_dict.get('fund_disbursement_percent', 10.0) or 10.0)
            protest_flag = bool(raw_dict.get('local_protest_flag', False))
            lapse_status = "Lapsed (Sec 19(7))" if sec11_days > 365 else ("Pre-Lapse Urgent (<90d)" if sec11_days >= 270 else "Compliant Active")
            days_to_lapse = max(0, 365 - sec11_days)

            item = {
                "project_id": proj_id,
                "project_name": proj_name,
                "state": state,
                "district": district,
                "project_type": project_type,
                "terrain_type": terrain,
                "latitude": lat,
                "longitude": lon,
                "status": status,
                "delay_probability": delay_prob,
                "risk_tier": tier_val,
                "composite_risk_score": round(crs_val, 1),
                "predicted_delay_days": predicted_delay_days,
                "land_area_hectares": land_ha,
                "estimated_cost_inr_crore": cost_cr,
                "section_11_notification_days": sec11_days,
                "compensation_multiplier_demand": comp_mult,
                "solatium_percentage": 100.0,
                "affected_families_count": aff_families,
                "title_dispute_rate_percent": dispute_pct,
                "sia_approval_status": sia_status,
                "forest_clearance_status": fc_status,
                "fund_disbursement_percent": fund_pct,
                "local_protest_flag": protest_flag,
                "larr_lapse_status": lapse_status,
                "larr_days_to_lapse": days_to_lapse
            }

            detail_item = dict(item)
            detail_item["median_survival_days"] = med_surv

            results.append(item)
            details_by_id[proj_id] = detail_item
            raw_rows_by_id[proj_id] = raw_dict

        elapsed = time.perf_counter() - start_time
        logging.info("Successfully processed and cached complete %d geo projects in %.2f seconds.", len(results), elapsed)

        _GEO_CACHE["data"] = results
        _GEO_CACHE["timestamp"] = time.time()
        _GEO_CACHE["details_by_id"] = details_by_id
        _GEO_CACHE["raw_rows_by_id"] = raw_rows_by_id

        if max_projects is not None:
            return results[:max_projects]
        return results

@app.post("/ai/advisory")
@limiter.limit("20/minute")
async def get_ai_advisory(request: Request, req: AIAdvisoryRequest, user: Any = Depends(get_current_user)):
    advisor = AIAdvisor()
    res = await run_in_threadpool(advisor.generate_advisory, req.query, req.context, req.project_metadata)
    return res

def generate_statutory_delay_explanation(
    payload: ProjectPayload,
    pred_days: int,
    prob_pct: float,
    crs_val: float,
    tier: str,
    milestones: List[Dict[str, Any]],
    sec11_days: int,
    lapse_triggered: bool,
    pre_lapse_urgent: bool,
    days_to_lapse: int
) -> Dict[str, Any]:
    project_name = payload.project_id or "Project"
    state = payload.state or "State"
    district = payload.district or "District"
    p_type = (payload.project_type or "Infrastructure").replace("_", " ")
    terrain = (payload.terrain_type or "General").replace("_", " ")
    cost = payload.estimated_cost_inr_crore or 0.0
    land = payload.land_area_hectares or 0.0
    pafs = payload.affected_families_count or 0
    comp_mult = payload.compensation_multiplier_demand or 1.0
    dispute_rate = payload.title_dispute_rate_percent or 0.0
    sia_status = payload.sia_approval_status or "Not_Required"
    fc_status = payload.forest_clearance_status or "Not_Required"
    fund_pct = payload.fund_disbursement_percent or 0.0
    protest = payload.local_protest_flag or False

    contributing_factors = []
    primary_bottleneck = ""
    critical_milestone = ""
    statutory_act = ""
    legal_hazard = ""

    # 1. Evaluate Critical Path & Statutory Bottlenecks
    if lapse_triggered:
        primary_bottleneck = "Statutory Section 19(7) Acquisition Lapse (Exceeded 365 Days)"
        critical_milestone = "Section 11 Notification Aging & Sec 19 Declaration"
        statutory_act = "RFCTLARR Act 2013, Section 19(7)"
        legal_hazard = f"Exceeded statutory 12-month limit ({sec11_days} days elapsed). Proceedings legally lapsed under Sec 19(7); fresh Section 11 gazette notification and socio-economic surveys required."
        contributing_factors.append({
            "label": "Sec 19(7) Clock Lapsed",
            "value": f"{sec11_days}d / 365d limit",
            "impact": "+140d statutory reset",
            "severity": "Critical"
        })
    elif pre_lapse_urgent:
        primary_bottleneck = f"Pre-Lapse Section 19 Declaration Deadline Imminent ({days_to_lapse} Days Left)"
        critical_milestone = "Section 19 Declaration Publication"
        statutory_act = "RFCTLARR Act 2013, Section 19(7)"
        legal_hazard = f"Only {days_to_lapse} days remaining in the 365-day statutory window. Failure to issue Section 19 declaration forces statutory lapse under Section 19(7)."
        contributing_factors.append({
            "label": "Imminent Statutory Lapse",
            "value": f"{days_to_lapse}d left",
            "impact": "+65d acceleration pressure",
            "severity": "High"
        })

    if fc_status in ["Pending", "Stage_1_Pending"]:
        severity = "High" if terrain in ["Forest Eco Sensitive", "Hilly", "Forest_Eco_Sensitive"] else "Medium"
        if not primary_bottleneck or terrain in ["Forest Eco Sensitive", "Forest_Eco_Sensitive"]:
            primary_bottleneck = "MoEF&CC Stage-1 Forest Clearance Deadlock on Parivesh"
            critical_milestone = "Forest & Environmental Clearances"
            statutory_act = "Forest (Conservation) Act 1980 & EIA 2006 Notification"
            legal_hazard = "Pending non-forest compensatory afforestation (CA) land mutation and tree enumeration NOCs on Parivesh portal hold back statutory possession."
        contributing_factors.append({
            "label": "Forest Clearance Status",
            "value": fc_status.replace("_", " "),
            "impact": "+50d clearance queue",
            "severity": severity
        })

    if pafs >= 1000:
        contributing_factors.append({
            "label": "Large-Scale Resettlement (PAFs)",
            "value": f"{pafs:,} Affected Families",
            "impact": "+45d Second Schedule R&R",
            "severity": "High" if pafs >= 3000 else "Medium"
        })

    if protest or comp_mult > 1.8:
        if not primary_bottleneck:
            primary_bottleneck = "Landowner Compensation Disparity & Public Resistance" if protest else "Landowner Compensation Multiplier Disparity"
            critical_milestone = "Compensation & Rehabilitation Settlement"
            statutory_act = "RFCTLARR Act 2013, Section 23 & 30 (Award & 100% Solatium)"
            legal_hazard = (
                f"Disparity in landowner expectations ({comp_mult:.2f}x multiplier demand) "
                f"{'coupled with active community protests' if protest else 'and compensation determination negotiations'} "
                f"prevent smooth disbursement of awards and halt Section 38 possession handover."
            )
        contributing_factors.append({
            "label": "Compensation Demand",
            "value": f"{comp_mult:.2f}x Multiplier",
            "impact": "+35d negotiation friction",
            "severity": "High" if comp_mult > 2.0 or protest else "Medium"
        })
        if protest:
            contributing_factors.append({
                "label": "Local Community Protests",
                "value": "Active Resistance Flagged",
                "impact": "+30d public hearing stalemate",
                "severity": "High"
            })

    if dispute_rate > 15.0:
        if not primary_bottleneck:
            primary_bottleneck = "Cadastral Title Disputes & LARRA Judicial References"
            critical_milestone = "Land Title Dispute Adjudication"
            statutory_act = "RFCTLARR Act 2013, Section 15 & 64 (Tribunal References)"
            legal_hazard = f"High title dispute rate ({dispute_rate:.1f}%) in {district} district triggers Section 15 objection hearings and statutory references to the LARRA Authority under Section 64."
        contributing_factors.append({
            "label": "Title Dispute Rate",
            "value": f"{dispute_rate:.1f}% contested parcels",
            "impact": f"+{round(dispute_rate * 2.5)}d judicial delay",
            "severity": "High" if dispute_rate > 25.0 else "Medium"
        })

    if sia_status == "Pending":
        if not primary_bottleneck:
            primary_bottleneck = "Social Impact Assessment (SIA) Appraisal Incomplete"
            critical_milestone = "Social Impact Assessment (SIA)"
            statutory_act = "RFCTLARR Act 2013, Section 4 & 7"
            legal_hazard = "Statutory Expert Group recommendation under Section 7 is pending, barring District Collector from proceeding with Section 11 gazette notification."
        contributing_factors.append({
            "label": "SIA Review Status",
            "value": "Pending Expert Appraisal",
            "impact": "+45d pre-notification freeze",
            "severity": "Medium"
        })

    if fund_pct < 40.0 and cost > 100.0:
        contributing_factors.append({
            "label": "Capital Disbursement",
            "value": f"{fund_pct:.1f}% disbursed",
            "impact": "Impairs 100% Solatium liquidity",
            "severity": "Medium" if fund_pct < 25.0 else "Low"
        })

    if not primary_bottleneck:
        primary_bottleneck = "Procedural Statutory Milestone Progression"
        critical_milestone = "Standard Administrative Cadastral Verification"
        statutory_act = "RFCTLARR Act 2013, Section 11 & 19"
        legal_hazard = "No statutory violations detected. Normal procedural timelines apply for revenue record verification."
        contributing_factors.append({
            "label": "Statutory Milestones",
            "value": "Compliant Schedule",
            "impact": "Nominal timeline",
            "severity": "Low"
        })

    # 2. Build Rich Contextual Narrative
    para1 = (
        f"For the **{project_name}** ({p_type}) spanning **{land:.1f} hectares** across **{district}, {state}** "
        f"({terrain} terrain), the predictive engine forecasts a delay probability of **{prob_pct:.1f}%** "
        f"with a timeline extension of **{pred_days} days** (Composite Risk Score: **{crs_val:.1f} / 100**, {tier} Risk). "
        f"The primary critical path bottleneck stalling project commissioning is **{primary_bottleneck}**."
    )

    p2_parts = []
    if lapse_triggered:
        p2_parts.append(
            f"Under **Section 19(7) of the RFCTLARR Act 2013**, preliminary notifications legally lapse if the Section 19 "
            f"declaration is not gazetted within 12 months. Having reached **{sec11_days} days** since preliminary publication, "
            f"the acquisition is statutorily deadlocked, risking voiding of previous socio-economic surveys and necessitating a fresh gazette notice."
        )
    elif pre_lapse_urgent:
        p2_parts.append(
            f"With **{sec11_days} days** elapsed and merely **{days_to_lapse} days remaining** before statutory lapse under Section 19(7), "
            f"the Competent Authority faces intense procedural compression to settle objections and issue the final Section 19 declaration."
        )

    if fc_status in ["Pending", "Stage_1_Pending"]:
        p2_parts.append(
            f"Environmental review under the **Forest (Conservation) Act 1980** remains active at Stage-1 on the MoEF&CC Parivesh portal. "
            f"Because the corridor traverses {terrain} terrain, non-forest compensatory afforestation demarcation and Gram Sabha clearances under FRA 2006 "
            f"represent mandatory pre-conditions before physical possession can be transferred to the executing agency."
        )

    if comp_mult > 1.8 or protest:
        p2_parts.append(
            f"Compensation expectations from **{pafs:,} affected families** stand elevated at **{comp_mult:.2f}x multiplier** (against statutory rural baseline). "
            f"{'Coupled with active grassroots protests, ' if protest else ''}"
            f"this creates substantial deadlock during Section 23 award determination and threatens voluntary handover under Section 38."
        )

    if pafs >= 2000:
        p2_parts.append(
            f"Additionally, the large displacement of **{pafs:,} Project-Affected Families (PAFs)** requires mandatory Rehabilitation & Resettlement schemes under the **Second Schedule of RFCTLARR Act 2013**, including Administrator for R&R appointment (Section 43) and formal rehabilitation colony site approvals."
        )

    if dispute_rate > 15.0:
        p2_parts.append(
            f"Furthermore, a **{dispute_rate:.1f}% title dispute rate** in {district} cadastral records indicates widespread co-tenancy and mutation conflicts, "
            f"leading to statutory reference petitions before the Land Acquisition Authority (LARRA) under Section 64 and prolonging award finalization."
        )

    if not p2_parts:
        p2_parts.append(
            f"The project maintains compliant statutory milestone velocity ({sec11_days}/365 days elapsed). "
            f"Clearances for SIA ({sia_status}) and Forest ({fc_status}) show minimal critical-path friction, keeping the corridor on-track."
        )

    para2 = " ".join(p2_parts)

    para3 = (
        f"**Statutory Resolution Strategy:** Mitigating this risk requires addressing **{statutory_act}**. "
        f"Prioritizing {'immediate Section 19 gazette declaration issuance' if (lapse_triggered or pre_lapse_urgent) else ''}"
        f"{', expedited Parivesh nodal file tracking' if fc_status in ['Pending', 'Stage_1_Pending'] else ''}"
        f"{', structured Gram Sabha ombudsman dialogue on compensation' if (comp_mult > 1.8 or protest) else ''}"
        f"{' and digital cadastral title reconciliation' if dispute_rate > 15.0 else ''} "
        f"can recover an estimated **{min(pred_days - 10, max(20, int(pred_days * 0.4)))} days** of statutory drift."
    )

    full_narrative = f"{para1}\n\n{para2}\n\n{para3}"

    return {
        "headline": f"{tier} Risk Delay Diagnosis: {primary_bottleneck}",
        "primary_bottleneck": primary_bottleneck,
        "critical_milestone": critical_milestone,
        "statutory_act": statutory_act,
        "legal_hazard": legal_hazard,
        "narrative": full_narrative,
        "paragraph_intro": para1,
        "paragraph_statutory": para2,
        "paragraph_strategy": para3,
        "contributing_factors": contributing_factors[:4],
        "schedule_impact": {
            "predicted_delay_days": pred_days,
            "delay_probability_pct": prob_pct,
            "composite_risk_score": crs_val,
            "risk_tier": tier
        }
    }

async def _execute_prediction_pipeline(payload: ProjectPayload) -> dict:
    if not system:
        raise HTTPException(status_code=500, detail="Models not loaded")

    # Security validation on free text / identifiers
    security_validator = PromptSecurityValidator()
    for field_val in [payload.project_id, payload.district]:
        if field_val:
            is_inj, reason = security_validator.detect_injection(str(field_val))
            if is_inj:
                raise HTTPException(status_code=400, detail=f"Security rejection: {reason}")

    payload_dict = payload.model_dump(exclude_unset=True) if hasattr(payload, 'model_dump') else payload.dict(exclude_unset=True)
    # Remove schedule-specific and geospatial metadata fields so they don't pollute the ML feature dataframe
    sched_tasks = payload_dict.pop('schedule_tasks', None)
    target_comp = payload_dict.pop('target_completion_days', None)
    geo_lat = payload_dict.pop('latitude', None)
    geo_lon = payload_dict.pop('longitude', None)
    geo_road = payload_dict.pop('road_type', None) or payload_dict.pop('road_connectivity_type', None)
    geo_addr = payload_dict.pop('address', None)

    # Remoteness & Urban-Tier Accessibility Evaluation
    remoteness_analysis = None
    try:
        query_addr = geo_addr
        if not query_addr and payload.district and str(payload.district).strip() not in ["", "Unknown", "nan"]:
            query_addr = f"{payload.district}, {payload.state}"

        eff_lat = geo_lat
        eff_lon = geo_lon
        if (eff_lat is None or eff_lon is None) and payload.state and payload.district:
            coords = get_district_coordinates(payload.state, payload.district)
            if coords:
                eff_lat, eff_lon = coords

        remoteness_analysis = evaluate_remoteness(
            lat=eff_lat,
            lon=eff_lon,
            address=query_addr,
            project_type=payload.project_type,
            district=payload.district,
            state=payload.state,
            provided_road_type=geo_road,
            provided_terrain=payload.terrain_type,
            allow_online=False
        )
    except Exception as re_err:
        logging.warning("Remoteness evaluation non-fatal error for %s: %s", payload.project_id, re_err)

    raw_payload = _prepare_df(payload_dict)

    metadata = {
        'project_id': payload.project_id,
        'estimated_cost_inr_crore': payload.estimated_cost_inr_crore,
        'terrain_type': payload.terrain_type,
        'sia_approval_status': payload.sia_approval_status,
        'forest_clearance_status': payload.forest_clearance_status,
        'title_dispute_rate_percent': payload.title_dispute_rate_percent,
        'local_protest_flag': payload.local_protest_flag,
        'fund_disbursement_percent': payload.fund_disbursement_percent,
        'section_11_notification_days': payload.section_11_notification_days,
        'schedule_tasks': payload.schedule_tasks,
        'target_completion_days': payload.target_completion_days
    }

    try:
        # Non-blocking threadpool offloading to preserve event loop concurrency
        result = await run_in_threadpool(system.predict, raw_payload, metadata=metadata)
        survival_curve = _extract_survival_curve(raw_payload)

        # Meta coefficients from StackingClassifier
        meta_coefs = {}
        if system.explainer and hasattr(system.explainer, 'meta_coefficients'):
            meta_coefs = {k: round(float(v), 3) for k, v in system.explainer.meta_coefficients.items()}

        # Top full features with signed TreeSHAP impacts
        feature_labels = {
            "F_r": "Fund Disbursement Risk Ratio (F_r)",
            "C_r": "Compensation Demand Ratio (C_r)",
            "P_r": "Protest & Agitation Risk Factor (P_r)",
            "H_r": "Historical State Delay Ratio (H_r)",
            "W_r": "Weather Vulnerability Index (W_r)",
            "affected_families_count": "Affected Families Count",
            "title_dispute_rate_percent": "Title Dispute Rate (%)",
            "local_protest_flag": "Local Agitation / Protest Flag",
            "compensation_multiplier_demand": "Compensation Multiplier Demand",
            "forest_clearance_status": "Forest Clearance Status",
            "forest_clearance_status_risk_score": "Forest Clearance Risk Score",
            "sia_approval_status": "SIA Approval Status",
            "sia_approval_status_risk_score": "SIA Approval Risk Score",
            "fund_disbursement_percent": "Fund Disbursement Progress (%)",
            "land_area_hectares": "Total Land Extent (Hectares)",
            "estimated_cost_inr_crore": "Estimated Capital Outlay (INR Cr)",
            "land_area_log": "Log Land Area",
            "project_age_years": "Elapsed Project Duration (Years)"
        }

        full_feats = []
        if 'local_explanation' in result.get('explanation', {}):
            for row in result['explanation']['local_explanation']:
                feat = row.get('feature')
                shap_val = float(row.get('shap_impact', 0.0))
                full_feats.append({
                    "feature": feat,
                    "feature_label": feature_labels.get(feat, feat.replace('_', ' ').title()),
                    "category": row.get('category', 'Operational'),
                    "shap_impact": round(shap_val, 4),
                    "impact_direction": "Increases Delay" if shap_val > 0 else "Decreases Delay",
                    "feature_value": row.get('feature_value')
                })

        full_feats_sorted = sorted(full_feats, key=lambda x: abs(x.get('shap_impact', 0)), reverse=True)[:10]

        top_drivers = result['explanation'].get('risk_drivers', [])
        for d in top_drivers:
            d['feature_label'] = feature_labels.get(d.get('feature'), d.get('feature', '').replace('_', ' ').title())

        # Prescriptive actions calculation
        prescriptive_actions = calculate_prescriptive_actions(result, payload.estimated_cost_inr_crore, raw_payload)

        # Statutory Milestone Breakdown under RFCTLARR Act 2013
        sec11_days = int(payload.section_11_notification_days or 30)
        statutory_limit = 365
        days_to_lapse = max(0, statutory_limit - sec11_days)
        lapse_triggered = sec11_days > statutory_limit
        pre_lapse_urgent = (sec11_days >= 270) and not lapse_triggered

        sia_delay = 45.0 if payload.sia_approval_status == 'Pending' else (60.0 if payload.sia_approval_status == 'Rejected' else 0.0)
        sec11_delay = max(0.0, (sec11_days - 180) * 0.4) if sec11_days > 180 else 0.0
        fc_delay = 50.0 if payload.forest_clearance_status in ['Pending', 'Stage_1_Pending'] else (80.0 if payload.forest_clearance_status == 'Rejected' else 0.0)
        dispute_delay = (payload.title_dispute_rate_percent or 0.0) * 2.5
        comp_delay = ((payload.compensation_multiplier_demand or 1.0) - 1.0) * 35.0
        protest_delay = 30.0 if payload.local_protest_flag else 0.0

        milestones = [
            {
                "milestone": "Social Impact Assessment (SIA)",
                "statutory_act": "RFCTLARR Act 2013 Sec 4 & 7",
                "status": payload.sia_approval_status or 'Pending',
                "estimated_delay_days": round(sia_delay, 1),
                "is_critical_path": sia_delay >= max(fc_delay, dispute_delay, comp_delay, protest_delay, sec11_delay)
            },
            {
                "milestone": "Section 11 Preliminary Notification",
                "statutory_act": "RFCTLARR Act 2013 Sec 11 & 19(7)",
                "days_elapsed": sec11_days,
                "statutory_limit_days": statutory_limit,
                "days_remaining_to_lapse": days_to_lapse,
                "lapse_warning": lapse_triggered,
                "pre_lapse_warning": pre_lapse_urgent,
                "estimated_delay_days": round(sec11_delay, 1),
                "is_critical_path": lapse_triggered or (sec11_delay >= max(sia_delay, fc_delay, dispute_delay))
            },
            {
                "milestone": "Forest & Environmental Clearances",
                "statutory_act": "Forest Conservation Act 1980",
                "status": payload.forest_clearance_status or 'Not_Required',
                "estimated_delay_days": round(fc_delay, 1),
                "is_critical_path": fc_delay >= max(sia_delay, dispute_delay, comp_delay, protest_delay, sec11_delay)
            },
            {
                "milestone": "Land Title Dispute Adjudication",
                "statutory_act": "RFCTLARR Act 2013 Sec 15 & 64 (LARRA)",
                "dispute_rate_pct": round(payload.title_dispute_rate_percent or 0.0, 1),
                "estimated_delay_days": round(dispute_delay, 1),
                "is_critical_path": dispute_delay >= max(sia_delay, fc_delay, comp_delay, protest_delay, sec11_delay)
            },
            {
                "milestone": "Compensation & Rehabilitation Settlement",
                "statutory_act": "RFCTLARR Act 2013 Sec 23, 26-30 (Award & 100% Solatium)",
                "estimated_delay_days": round(comp_delay + protest_delay, 1),
                "is_critical_path": (comp_delay + protest_delay) >= max(sia_delay, fc_delay, dispute_delay, sec11_delay)
            }
        ]

        pred_days_val = int(result['predictions']['predicted_delay_days'])
        months_val = round(pred_days_val / 30.4375, 1)
        weeks_val = int(round(pred_days_val / 7.0))
        delay_human = f"~{months_val} Months ({weeks_val} Weeks)" if pred_days_val >= 30 else f"~{weeks_val} Weeks ({pred_days_val} Days)"
        
        prob_val = round(result['predictions']['delay_probability'] * 100, 1)
        conf_score = round(max(prob_val, 100.0 - prob_val), 1)
        conf_label = "High Certainty" if conf_score >= 80 else ("Moderate Certainty" if conf_score >= 65 else "Low Certainty")

        crs_val = round(float(result['predictions'].get('crs', 0.0)), 1)
        calibrated_tier = "High" if crs_val > 50.0 else ("Medium" if crs_val > 25.0 else "Low")

        ai_delay_explanation = generate_statutory_delay_explanation(
            payload=payload,
            pred_days=pred_days_val,
            prob_pct=prob_val,
            crs_val=crs_val,
            tier=calibrated_tier,
            milestones=milestones,
            sec11_days=sec11_days,
            lapse_triggered=lapse_triggered,
            pre_lapse_urgent=pre_lapse_urgent,
            days_to_lapse=days_to_lapse
        )

        # Map to Frontend Schema
        frontend_response = {
            "project_id": payload.project_id,
            "ai_delay_explanation": ai_delay_explanation,
            "predictions": {
                "delay_probability": prob_val,
                "confidence_score": conf_score,
                "confidence_label": conf_label,
                "calibrated_risk_tier": calibrated_tier,
                "predicted_delay_days": pred_days_val,
                "delay_human_readable": delay_human,
                "delay_months": months_val,
                "delay_weeks": weeks_val,
                "error_margin_days_mae": 31.58,
                "error_margin_days_conformal": 65.23,
                "error_margin_crs_mae": 0.047,
                "error_margin_crs_conformal": 0.097,
                "median_survival_days": int(result['timeline']['median_survival_days']),
                "crs": crs_val,
                "days_p10": round(float(result['predictions'].get('days_p10', pred_days_val - 65)), 1),
                "days_p90": round(float(result['predictions'].get('days_p90', pred_days_val + 65)), 1),
                "crs_p10": round(float(result['predictions'].get('crs_p10', result['predictions']['crs'] - 0.1)), 1),
                "crs_p90": round(float(result['predictions'].get('crs_p90', result['predictions']['crs'] + 0.1)), 1),
                "adjusted_risk_index": round(float(result['predictions'].get('adjusted_risk_index', result['predictions']['crs'])), 1),
                "risk_phase": result['timeline'].get('risk_phase', 'Short-term'),
                "predicted_delay_rationale": result['predictions'].get('predicted_delay_rationale', ''),
                "uno_c_index": 0.906,
                "c_index_str": "0.9060 ± 0.0020"
            },
            "model_accuracy": {
                "uno_c_index": 0.906,
                "c_index_ci": "0.9060 ± 0.0020",
                "timeline_r2": 0.9460,
                "timeline_mae_days": 31.58,
                "timeline_rmse_days": 40.84,
                "timeline_mape_pct": 11.33,
                "crs_r2": 0.99997,
                "crs_mae": 0.047,
                "crs_mape_pct": 0.09,
                "classification_accuracy": 87.22,
                "classification_roc_auc": 0.9402,
                "classification_f1": 0.8417,
                "conformal_coverage_pct": 90.01
            },
            "timeline": {
                "c_index": 0.906,
                "c_index_str": "0.9060 ± 0.0020",
                "median_survival_days": int(result['timeline']['median_survival_days']),
                "risk_phase": result['timeline'].get('risk_phase', 'Short-term')
            },
            "larr_compliance": {
                "section_11_notification_days": sec11_days,
                "statutory_limit_days": statutory_limit,
                "days_to_lapse": days_to_lapse,
                "lapse_status": "Lapsed (Sec 19(7))" if lapse_triggered else ("Pre-Lapse Urgent (<90d)" if pre_lapse_urgent else "Compliant Active"),
                "statutory_lapse_warning": lapse_triggered,
                "pre_lapse_warning": pre_lapse_urgent,
                "compensation_multiplier": payload.compensation_multiplier_demand,
                "solatium_percentage": 100.0,
                "solatium_act": "RFCTLARR Act 2013 Sec 30 (100% Mandatory Solatium)",
                "affected_families_count": payload.affected_families_count,
                "rr_act": "RFCTLARR Act 2013 Second Schedule (R&R Entitlements)",
                "sia_status": payload.sia_approval_status,
                "sia_act": "RFCTLARR Act 2013 Sec 4 & 7",
                "title_dispute_rate_percent": payload.title_dispute_rate_percent,
                "dispute_act": "RFCTLARR Act 2013 Sec 15 & 64 (LARRA Authority)"
            },
            "milestones": milestones,
            "explainability": {
                "top_risk_drivers": result['explanation']['risk_drivers'],
                "category_breakdown": result['explanation']['category_breakdown'],
                "local_explanation_full": full_feats_sorted,
                "meta_coefficients": meta_coefs,
                "global_importance": result['explanation'].get('global_importance_approx', [])[:8]
            },
            "survival_curve": survival_curve,
            "remoteness_analysis": remoteness_analysis,
            "recommendations": prescriptive_actions,
            "prescriptive_actions": prescriptive_actions
        }
        return frontend_response
    except Exception as e:
        logging.error("Inference pipeline failed for project %s: %s", payload.project_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/predict")
@limiter.limit("60/minute")
async def predict_risk(request: Request, payload: ProjectPayload, user: Any = Depends(get_current_user)):
    return await _execute_prediction_pipeline(payload)

@app.post("/remoteness/evaluate")
@limiter.limit("120/minute")
async def evaluate_site_remoteness(
    request: Request,
    payload: RemotenessRequest
):
    """
    Evaluates physical remoteness and urban-tier accessibility delay for a project site.
    Returns:
    - nearest settlement (name, tier, distance km, Census vintage)
    - road connectivity classification
    - terrain type & Forest/Tribal status
    - estimated remoteness-driven delay days
    - normalized Remoteness Score (0-1)
    - component breakdown for explainability
    - data quality flags
    """
    try:
        eff_lat = payload.latitude
        eff_lon = payload.longitude
        if (eff_lat is None or eff_lon is None) and payload.state and payload.district:
            coords = get_district_coordinates(payload.state, payload.district)
            if coords:
                eff_lat, eff_lon = coords

        res = evaluate_remoteness(
            lat=eff_lat,
            lon=eff_lon,
            address=payload.address or (f"{payload.district}, {payload.state}" if payload.district and payload.district != "Unknown" else None),
            project_type=payload.project_type,
            district=payload.district,
            state=payload.state,
            provided_road_type=payload.road_type,
            provided_terrain=payload.terrain_type,
            allow_online=bool(payload.allow_online)
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error("Remoteness evaluation error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Remoteness evaluation failed: {e}")

# --- Phase 10: Persistent Memory Endpoints ---
@app.post("/analyses/save")
@limiter.limit("60/minute")
async def save_analysis(request: Request, req: SaveAnalysisRequest, user: Any = Depends(get_current_user)):
    """
    Saves a completed risk analysis with persistent memory in SQLite.
    Computes/verifies prediction, derives deterministic coordinates based on UUID seed,
    and returns the stored record.
    """
    if not req.project_name or not req.project_name.strip():
        raise HTTPException(status_code=400, detail="project_name is required and cannot be empty")

    if not isinstance(req.input_payload, dict):
        raise HTTPException(status_code=400, detail="input_payload must be a JSON object")

    try:
        payload_obj = ProjectPayload(**req.input_payload)
    except Exception as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input_payload: {ve}")

    frontend_response = await _execute_prediction_pipeline(payload_obj)

    state = (req.state or payload_obj.state or req.input_payload.get('state') or 'Unknown').strip()
    district = (req.district or payload_obj.district or req.input_payload.get('district') or 'Unknown').strip()
    project_type = (payload_obj.project_type or req.input_payload.get('project_type') or 'Infrastructure').strip()

    rec_id = str(uuid.uuid4())
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    created_by_email = user.get("email", "unknown@user")

    # Jitter seed derived deterministically from the record UUID
    jitter_seed = (uuid.UUID(rec_id).int % 30) + 1
    lat, lon = derive_coordinates({}, rec_id, state, district, intra_index=jitter_seed)

    preds = frontend_response.get('predictions', {})
    delay_prob = float(preds.get('delay_probability', 0.0))
    crs = float(preds.get('crs', 0.0))
    raw_tier = str(preds.get('calibrated_risk_tier', 'Medium'))
    if crs > 0:
        risk_tier = "High" if crs > 50.0 else ("Medium" if crs > 25.0 else "Low")
    else:
        risk_tier = "High" if raw_tier in ["Critical", "Very_High", "High"] else ("Medium" if raw_tier in ["Medium", "Moderate"] else "Low")
    pred_delay_days = int(preds.get('predicted_delay_days', 0))

    input_payload_json = json.dumps(req.input_payload)

    with _SAVED_DB_LOCK:
        conn = get_saved_db_connection()
        try:
            with conn:
                conn.execute("""
                    INSERT INTO saved_analyses (
                        id, project_name, created_by_email, state, district, project_type,
                        latitude, longitude, input_payload, delay_probability, risk_tier,
                        predicted_delay_days, composite_risk_score, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rec_id, req.project_name.strip(), created_by_email, state, district, project_type,
                    lat, lon, input_payload_json, delay_prob, risk_tier,
                    pred_delay_days, crs, created_at
                ))
        finally:
            conn.close()

    # --- Phase 11: Continuous Learning & Auto-Retraining ---
    # Automatically append newly saved project to training data store & trigger continuous retraining
    training_data_count = None
    try:
        from continuous_learning import ingest_new_projects, retrain_pipeline
        training_record = {
            "project_id": str(payload_obj.project_id or req.project_name.strip() or f"PROJ-{rec_id[:8]}"),
            "state": state,
            "district": district,
            "project_type": project_type,
            "terrain_type": getattr(payload_obj, 'terrain_type', None) or req.input_payload.get('terrain_type', 'Plain'),
            "estimated_cost_inr_crore": float(getattr(payload_obj, 'estimated_cost_inr_crore', 100.0) or 100.0),
            "land_area_hectares": float(getattr(payload_obj, 'land_area_hectares', 50.0) or 50.0),
            "sia_approval_status": getattr(payload_obj, 'sia_approval_status', 'Pending') or 'Pending',
            "forest_clearance_status": getattr(payload_obj, 'forest_clearance_status', 'Not_Required') or 'Not_Required',
            "fund_disbursement_percent": float(getattr(payload_obj, 'fund_disbursement_percent', 10.0) or 10.0),
            "affected_families_count": int(getattr(payload_obj, 'affected_families_count', 500) or 500),
            "title_dispute_rate_percent": float(getattr(payload_obj, 'title_dispute_rate_percent', 5.0) or 5.0),
            "compensation_multiplier_demand": float(getattr(payload_obj, 'compensation_multiplier_demand', 1.5) or 1.5),
            "section_11_notification_days": int(getattr(payload_obj, 'section_11_notification_days', 30) or 30),
            "local_protest_flag": bool(getattr(payload_obj, 'local_protest_flag', False)),
            "delay_binary_label": 1 if delay_prob >= 0.5 else 0,
            "delay_risk_tier": risk_tier,
            "Actual_Delay_Days": pred_delay_days,
            "CRS": crs,
            "CRS_tier": risk_tier
        }

        ingest_res = ingest_new_projects([training_record])
        training_data_count = ingest_res.get("data_store_count")
        logging.info(f"[ContinuousLearning] Saved project '{req.project_name.strip()}' ingested into data store. Total records: {training_data_count}")

        # Launch automated background retraining thread to continuously adapt model weights
        def _bg_continuous_retrain():
            global system
            try:
                logging.info(f"[ContinuousLearning] Starting continuous model retraining cycle for saved project: {req.project_name.strip()} (ID: {rec_id})...")
                res = retrain_pipeline(trigger_reason=f"db_save_{rec_id}")
                if res.get("promoted", False):
                    try:
                        system = RiskAnalysisSystem(
                            pipeline_path='pipeline.joblib',
                            ensemble_path='ensemble.joblib',
                            timeline_path='models_new/timeline.joblib' if os.path.exists('models_new/timeline.joblib') and os.path.getsize('models_new/timeline.joblib') > 1000 else 'timeline.joblib'
                        )
                        logging.info("[RELOADED] Hot-reloaded newly retrained continuous learning model weights into active API serving.")
                    except Exception as hot_err:
                        logging.warning(f"Note on continuous learning hot-reload: {hot_err}")
                logging.info(f"[ContinuousLearning] Retraining completed. Promoted: {res.get('promoted')}, Version: {res.get('version')}")
            except Exception as bg_err:
                logging.error(f"[ContinuousLearning] Continuous retraining background cycle failed: {bg_err}", exc_info=True)

        threading.Thread(target=_bg_continuous_retrain, daemon=True).start()
    except Exception as ing_err:
        logging.error(f"Failed to ingest saved project into continuous learning store: {ing_err}", exc_info=True)

    return {
        "id": rec_id,
        "project_name": req.project_name.strip(),
        "state": state,
        "district": district,
        "project_type": project_type,
        "latitude": lat,
        "longitude": lon,
        "delay_probability": delay_prob,
        "risk_tier": risk_tier,
        "predicted_delay_days": pred_delay_days,
        "composite_risk_score": crs,
        "created_at": created_at,
        "database": "saved_analyses.db",
        "continuous_learning": {
            "status": "ingested_and_retraining_triggered",
            "data_store_count": training_data_count
        }
    }

@app.post("/projects/save")
@limiter.limit("60/minute")
async def save_project(request: Request, req: SaveAnalysisRequest, user: Any = Depends(get_current_user)):
    """
    Alias for saving a project to SQLite database with continuous learning ingestion.
    """
    return await save_analysis(request, req, user)

@app.get("/analyses")
@limiter.limit("60/minute")
async def get_saved_analyses(request: Request, user: Any = Depends(get_current_user)):
    """
    Returns all saved analyses created by the authenticated user, ordered most recent first.
    """
    user_email = user.get("email", "")
    with _SAVED_DB_LOCK:
        conn = get_saved_db_connection()
        try:
            cur = conn.execute("""
                SELECT id, project_name, state, district, project_type,
                       latitude, longitude, input_payload, delay_probability, risk_tier,
                       predicted_delay_days, composite_risk_score, created_at
                FROM saved_analyses
                WHERE created_by_email = ?
                ORDER BY created_at DESC
            """, (user_email,))
            rows = cur.fetchall()
            results = []
            for r in rows:
                row_dict = dict(r)
                if row_dict.get("input_payload"):
                    try:
                        parsed = json.loads(row_dict["input_payload"]) if isinstance(row_dict["input_payload"], str) else row_dict["input_payload"]
                        row_dict["input_payload"] = parsed
                        if "terrain_type" in parsed:
                            row_dict["terrain_type"] = normalize_terrain_type(parsed["terrain_type"])
                    except Exception:
                        pass
                results.append(row_dict)
            return results
        finally:
            conn.close()

@app.get("/analyses/{id}")
@limiter.limit("60/minute")
async def get_saved_analysis_detail(request: Request, id: str, user: Any = Depends(get_current_user)):
    """
    Returns detail for one saved analysis including parsed input_payload.
    Rejects with 404 if not found or created by another user.
    """
    user_email = user.get("email", "")
    with _SAVED_DB_LOCK:
        conn = get_saved_db_connection()
        try:
            cur = conn.execute("""
                SELECT id, project_name, created_by_email, state, district, project_type,
                       latitude, longitude, input_payload, delay_probability, risk_tier,
                       predicted_delay_days, composite_risk_score, created_at
                FROM saved_analyses
                WHERE id = ?
            """, (id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Analysis not found")
            row_dict = dict(row)
            if row_dict.get("created_by_email") != user_email:
                raise HTTPException(status_code=404, detail="Analysis not found")

            try:
                parsed = json.loads(row_dict["input_payload"]) if isinstance(row_dict["input_payload"], str) else row_dict["input_payload"]
                row_dict["input_payload"] = parsed
                if "terrain_type" in parsed:
                    row_dict["terrain_type"] = normalize_terrain_type(parsed["terrain_type"])
            except Exception:
                pass
            row_dict.pop("created_by_email", None)
            return row_dict
        finally:
            conn.close()

@app.delete("/analyses/{id}")
@limiter.limit("60/minute")
async def delete_saved_analysis(request: Request, id: str, user: Any = Depends(get_current_user)):
    """
    Deletes the saved analysis record if owned by the requesting user.
    Returns 404 otherwise.
    """
    user_email = user.get("email", "")
    with _SAVED_DB_LOCK:
        conn = get_saved_db_connection()
        try:
            with conn:
                cur = conn.execute("""
                    DELETE FROM saved_analyses
                    WHERE id = ? AND created_by_email = ?
                """, (id, user_email))
                if cur.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Analysis not found")
            return {"status": "deleted", "id": id}
        finally:
            conn.close()

# --- State-to-Districts Reference Mapping Cache ---
_DISTRICTS_MAPPING_CACHE: Optional[Dict[str, List[str]]] = None
_DISTRICTS_MAPPING_LOCK = threading.Lock()

# Official 75 Districts of Uttar Pradesh
UP_ALL_75_DISTRICTS = [
    "Agra", "Aligarh", "Ambedkar Nagar", "Amethi", "Amroha", "Auraiya", "Ayodhya", "Azamgarh",
    "Baghpat", "Bahraich", "Ballia", "Balrampur", "Banda", "Barabanki", "Bareilly", "Basti",
    "Bhadohi", "Bijnor", "Budaun", "Bulandshahr", "Chandauli", "Chitrakoot", "Deoria", "Etah",
    "Etawah", "Farrukhabad", "Fatehpur", "Firozabad", "Gautam Buddha Nagar (Noida)", "Ghaziabad",
    "Ghazipur", "Gonda", "Gorakhpur", "Hamirpur", "Hapur", "Hardoi", "Hathras", "Jalaun",
    "Jaunpur", "Jhansi", "Kannauj", "Kanpur Dehat", "Kanpur Nagar", "Kasganj", "Kaushambi",
    "Kushinagar", "Lakhimpur Kheri", "Lalitpur", "Lucknow", "Maharajganj", "Mahoba", "Mainpuri",
    "Mathura", "Mau", "Meerut", "Mirzapur", "Moradabad", "Muzaffarnagar", "Pilibhit", "Pratapgarh",
    "Prayagraj", "Raebareli", "Rampur", "Saharanpur", "Sambhal", "Sant Kabir Nagar", "Shahjahanpur",
    "Shamli", "Shravasti", "Siddharthnagar", "Sitapur", "Sonbhadra", "Sultanpur", "Unnao", "Varanasi"
]

def get_state_districts_mapping() -> Dict[str, List[str]]:
    global _DISTRICTS_MAPPING_CACHE
    if _DISTRICTS_MAPPING_CACHE is not None:
        return _DISTRICTS_MAPPING_CACHE

    with _DISTRICTS_MAPPING_LOCK:
        if _DISTRICTS_MAPPING_CACHE is not None:
            return _DISTRICTS_MAPPING_CACHE

        csv_path = "indian_infrastructure_projects_dataset.csv"
        if not os.path.exists(csv_path):
            if os.path.exists("Revolution-main/indian_infrastructure_projects_dataset.csv"):
                csv_path = "Revolution-main/indian_infrastructure_projects_dataset.csv"

        mapping: Dict[str, List[str]] = {}
        if os.path.exists(csv_path):
            try:
                # Read the FULL CSV file independently (no row cap, completely separate from /projects/geo)
                df = pd.read_csv(csv_path, usecols=['state', 'district'])
                total_rows = len(df)
                raw_shape = df.shape
                print(f"[Reference Districts] Reading FULL CSV: {total_rows} rows (shape: {raw_shape}), covering {df['state'].nunique()} unique states.")
                logging.info(
                    "[Reference Districts] Successfully read FULL CSV file '%s': %d rows (shape: %s), covering %d states.",
                    csv_path, total_rows, str(raw_shape), df['state'].nunique()
                )

                raw_state_counts = {}
                for state, group in df.groupby('state'):
                    state_str = str(state).strip()
                    districts = sorted(list(set([
                        str(d).strip() for d in group['district'].dropna().unique() 
                        if str(d).strip() 
                        and str(d).strip().lower() not in ['nan', 'none', 'unknown', '', 'null', 'undefined']
                        and not re.search(r'district\s+\d+', str(d), re.IGNORECASE)
                    ])))
                    mapping[state_str] = districts
                    raw_state_counts[state_str] = len(districts)

                # Complete official 75 districts for Uttar Pradesh (source CSV contains 25 infrastructure project districts;
                # augmented with all 75 official districts to provide complete reference dropdown coverage)
                up_csv_count = len(mapping.get("Uttar Pradesh", []))
                up_existing = set(d for d in mapping.get("Uttar Pradesh", []) if not re.search(r'district\s+\d+', d, re.IGNORECASE))
                up_combined = sorted(list(up_existing.union(set(UP_ALL_75_DISTRICTS))))
                mapping["Uttar Pradesh"] = up_combined

                print(f"[Reference Districts] Total unique districts per state from CSV (all {len(mapping)} states):")
                for s_name in sorted(mapping.keys()):
                    final_cnt = len(mapping[s_name])
                    raw_cnt = raw_state_counts.get(s_name, 0)
                    if s_name == "Uttar Pradesh":
                        print(f"  - {s_name}: {final_cnt} districts (augmented from {raw_cnt} in CSV to full {final_cnt} official districts)")
                    else:
                        print(f"  - {s_name}: {final_cnt} districts")

                logging.info(
                    "[Reference Districts] Mapping initialized for %d states. Uttar Pradesh: %d districts (raw CSV: %d -> full: 75), Rajasthan: %d, West Bengal: %d, Maharashtra: %d. Total CSV rows: %d.",
                    len(mapping), len(mapping.get("Uttar Pradesh", [])), up_csv_count, len(mapping.get("Rajasthan", [])),
                    len(mapping.get("West Bengal", [])), len(mapping.get("Maharashtra", [])), total_rows
                )
            except Exception as e:
                print(f"[Reference Districts] ERROR reading districts mapping from {csv_path}: {e}")
                logging.error("Failed reading districts mapping from %s: %s", csv_path, e)

        _DISTRICTS_MAPPING_CACHE = mapping
        return _DISTRICTS_MAPPING_CACHE

@app.get("/reference/districts")
async def get_reference_districts(request: Request):
    """
    Returns reference state-to-districts mapping computed once from the infrastructure dataset.
    """
    mapping = get_state_districts_mapping()
    return mapping

@app.get("/reference/coordinates")
async def get_reference_coordinates(request: Request):
    """
    Returns official reference coordinates map for all Indian districts from district_coordinates.json.
    """
    global _DISTRICT_COORDS
    if not _DISTRICT_COORDS:
        get_district_coordinates("Rajasthan", "Banswara")
    return _DISTRICT_COORDS

@app.get("/projects/geo")
@limiter.limit("120/minute")
async def get_projects_geo(
    request: Request,
    limit: Optional[int] = Query(200, description="Max projects to return (0 or >=13532 returns all 13,532)"),
    state: Optional[str] = Query(None, description="Filter by state name"),
    district: Optional[str] = Query(None, description="Filter by district name"),
    risk_tier: Optional[str] = Query(None, description="Filter by risk tier (Low, Medium, High)"),
    project_type: Optional[str] = Query(None, description="Filter by project type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Filter by search query"),
    user: Any = Depends(get_current_user)
):
    """
    Returns geographical distribution of infrastructure projects with calibrated risk predictions.
    Supports complete dataset (13,532 projects) across all 36 Indian States and 453 Districts.
    """
    try:
        data = await run_in_threadpool(get_or_load_geo_cache, max_projects=None)
        filtered = data
        if state:
            s_norm = state.strip().lower()
            filtered = [p for p in filtered if p['state'].lower() == s_norm or s_norm in p['state'].lower()]
        if district:
            d_norm = district.strip().lower()
            filtered = [p for p in filtered if p['district'].lower() == d_norm or d_norm in p['district'].lower()]
        if risk_tier:
            r_norm = risk_tier.strip().lower()
            filtered = [p for p in filtered if p['risk_tier'].lower() == r_norm]
        if project_type:
            pt_norm = project_type.strip().lower()
            filtered = [p for p in filtered if p['project_type'].lower() == pt_norm]
        if status:
            st_norm = status.strip().lower()
            filtered = [p for p in filtered if p['status'].lower() == st_norm]
        if search:
            q_norm = search.strip().lower()
            filtered = [p for p in filtered if q_norm in p['project_name'].lower() or q_norm in p['project_id'].lower() or q_norm in p['district'].lower() or q_norm in p['state'].lower()]

        if limit is not None and limit > 0:
            return filtered[:limit]
        return filtered
    except HTTPException:
        raise
    except Exception as e:
        logging.error("Failed to generate geo project collection: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate geo project collection: {e}")

@app.get("/projects/geo/{project_id}")
@limiter.limit("60/minute")
async def get_project_geo_detail(request: Request, project_id: str, user: Any = Depends(get_current_user)):
    """
    Returns full details for a single project including explainability risk drivers and prescriptive mitigations.
    Computed lazily on-demand the first time requested, then cached in memory.
    """
    if _GEO_CACHE["data"] is None:
        await run_in_threadpool(get_or_load_geo_cache, max_projects=None)

    cached_detail = _GEO_CACHE["details_by_id"].get(project_id)
    if not cached_detail:
        raise HTTPException(status_code=404, detail=f"Project with ID '{project_id}' not found.")

    if "explainability" not in cached_detail:
        raw_row = _GEO_CACHE["raw_rows_by_id"].get(project_id, {})
        try:
            raw_dict = dict(raw_row)
            raw_payload = _prepare_df(dict(raw_dict))
            metadata = {
                'project_id': project_id,
                'estimated_cost_inr_crore': float(raw_dict.get('estimated_cost_inr_crore', 100.0) or 100.0),
                'terrain_type': normalize_terrain_type(raw_dict.get('terrain_type') or raw_dict.get('Terrain_Type') or 'Rural_Agri'),
                'sia_approval_status': str(raw_dict.get('sia_approval_status', 'Pending')),
                'forest_clearance_status': str(raw_dict.get('forest_clearance_status', 'Not_Required')),
                'title_dispute_rate_percent': float(raw_dict.get('title_dispute_rate_percent', 5.0) or 5.0),
                'local_protest_flag': bool(raw_dict.get('local_protest_flag', False)),
                'fund_disbursement_percent': float(raw_dict.get('fund_disbursement_percent', 10.0) or 10.0),
                'section_11_notification_days': raw_dict.get('section_11_notification_days', 30)
            }
            full_res = await run_in_threadpool(system.predict, raw_payload, metadata=metadata)
            cost_cr = float(raw_dict.get('estimated_cost_inr_crore', 100.0) or 100.0)
            prescriptive_actions = calculate_prescriptive_actions(full_res, cost_cr, raw_payload)

            sec11_days = int(raw_dict.get('section_11_notification_days', 30) or 30)
            statutory_limit = 365
            days_to_lapse = max(0, statutory_limit - sec11_days)
            lapse_triggered = sec11_days > statutory_limit
            pre_lapse_urgent = (sec11_days >= 270) and not lapse_triggered
            sia_status = str(raw_dict.get('sia_approval_status', 'Pending')).strip()
            fc_status = str(raw_dict.get('forest_clearance_status', 'Not_Required')).strip()
            dispute_pct = float(raw_dict.get('title_dispute_rate_percent', 0.0) or 0.0)
            comp_mult = float(raw_dict.get('compensation_multiplier_demand', 1.5) or 1.5)
            aff_families = int(raw_dict.get('affected_families_count', 0) or 0)
            protest_flag = bool(raw_dict.get('local_protest_flag', False))

            sia_delay = 45.0 if sia_status == 'Pending' else (60.0 if sia_status == 'Rejected' else 0.0)
            sec11_delay = max(0.0, (sec11_days - 180) * 0.4) if sec11_days > 180 else 0.0
            fc_delay = 50.0 if fc_status in ['Pending', 'Stage_1_Pending'] else (80.0 if fc_status == 'Rejected' else 0.0)
            dispute_delay = dispute_pct * 2.5
            comp_delay = (comp_mult - 1.0) * 35.0
            protest_delay = 30.0 if protest_flag else 0.0

            milestones = [
                {
                    "milestone": "Social Impact Assessment (SIA)",
                    "statutory_act": "RFCTLARR Act 2013 Sec 4 & 7",
                    "status": sia_status,
                    "estimated_delay_days": round(sia_delay, 1),
                    "is_critical_path": sia_delay >= max(fc_delay, dispute_delay, comp_delay, protest_delay, sec11_delay)
                },
                {
                    "milestone": "Section 11 Preliminary Notification",
                    "statutory_act": "RFCTLARR Act 2013 Sec 11 & 19(7)",
                    "days_elapsed": sec11_days,
                    "statutory_limit_days": statutory_limit,
                    "days_remaining_to_lapse": days_to_lapse,
                    "lapse_warning": lapse_triggered,
                    "pre_lapse_warning": pre_lapse_urgent,
                    "estimated_delay_days": round(sec11_delay, 1),
                    "is_critical_path": lapse_triggered or (sec11_delay >= max(sia_delay, fc_delay, dispute_delay))
                },
                {
                    "milestone": "Forest & Environmental Clearances",
                    "statutory_act": "Forest Conservation Act 1980",
                    "status": fc_status,
                    "estimated_delay_days": round(fc_delay, 1),
                    "is_critical_path": fc_delay >= max(sia_delay, dispute_delay, comp_delay, protest_delay, sec11_delay)
                },
                {
                    "milestone": "Land Title Dispute Adjudication",
                    "statutory_act": "RFCTLARR Act 2013 Sec 15 & 64 (LARRA)",
                    "dispute_rate_pct": round(dispute_pct, 1),
                    "estimated_delay_days": round(dispute_delay, 1),
                    "is_critical_path": dispute_delay >= max(sia_delay, fc_delay, comp_delay, protest_delay, sec11_delay)
                },
                {
                    "milestone": "Compensation & Rehabilitation Settlement",
                    "statutory_act": "RFCTLARR Act 2013 Sec 23, 26-30 (Award & 100% Solatium)",
                    "estimated_delay_days": round(comp_delay + protest_delay, 1),
                    "is_critical_path": (comp_delay + protest_delay) >= max(sia_delay, fc_delay, dispute_delay, sec11_delay)
                }
            ]

            cached_detail["explainability"] = {
                "top_risk_drivers": full_res['explanation'].get('risk_drivers', []),
                "category_breakdown": full_res['explanation'].get('category_breakdown', {})
            }
            cached_detail["prescriptive_actions"] = prescriptive_actions
            cached_detail["recommendations"] = prescriptive_actions
            cached_detail["milestones"] = milestones
            cached_detail["larr_compliance"] = {
                "section_11_notification_days": sec11_days,
                "statutory_limit_days": statutory_limit,
                "days_to_lapse": days_to_lapse,
                "lapse_status": "Lapsed (Sec 19(7))" if lapse_triggered else ("Pre-Lapse Urgent (<90d)" if pre_lapse_urgent else "Compliant Active"),
                "statutory_lapse_warning": lapse_triggered,
                "pre_lapse_warning": pre_lapse_urgent,
                "compensation_multiplier": comp_mult,
                "solatium_percentage": 100.0,
                "solatium_act": "RFCTLARR Act 2013 Sec 30 (100% Mandatory Solatium)",
                "affected_families_count": aff_families,
                "rr_act": "RFCTLARR Act 2013 Second Schedule (R&R Entitlements)",
                "sia_status": sia_status,
                "sia_act": "RFCTLARR Act 2013 Sec 4 & 7",
                "title_dispute_rate_percent": dispute_pct,
                "dispute_act": "RFCTLARR Act 2013 Sec 15 & 64 (LARRA Authority)"
            }
            _GEO_CACHE["details_by_id"][project_id] = cached_detail
        except Exception as ex:
            logging.warning("Detailed explainability generation failed for %s: %s", project_id, ex)

    return cached_detail

@app.post("/simulate")
@limiter.limit("60/minute")
async def simulate_intervention(request: Request, payload: SimulationPayload, user: Any = Depends(get_current_user)):
    if not system:
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        # 1. Baseline
        base_dict = payload.baseline.dict(exclude_unset=True)
        base_df = _prepare_df(base_dict)
        base_res = system.predict(base_df)

        # 2. Modified with interventions
        mod_dict = dict(base_dict)
        mod_dict.update(payload.interventions)
        mod_df = _prepare_df(mod_dict)
        mod_res = system.predict(mod_df)

        base_prob = round(base_res['predictions']['delay_probability'] * 100, 1)
        mod_prob = round(mod_res['predictions']['delay_probability'] * 100, 1)
        base_days = int(base_res['predictions']['predicted_delay_days'])
        mod_days = int(mod_res['predictions']['predicted_delay_days'])

        delta_days = base_days - mod_days  # positive = saved
        delta_prob = round(base_prob - mod_prob, 1)  # positive = reduced risk

        return {
            "baseline": {
                "delay_probability": base_prob,
                "predicted_delay_days": base_days,
                "risk_tier": base_res['predictions']['calibrated_risk_tier']
            },
            "simulated": {
                "delay_probability": mod_prob,
                "predicted_delay_days": mod_days,
                "risk_tier": mod_res['predictions']['calibrated_risk_tier']
            },
            "impact": {
                "days_saved": max(0, delta_days),
                "prob_reduction_percent": max(0.0, delta_prob),
                "status": "Improved" if (delta_days > 0 or delta_prob > 0) else "Neutral"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")

@app.get("/metrics")
@limiter.limit("30/minute")
async def get_metrics(request: Request, user: Any = Depends(get_current_user)):
    """Phase 7: Monitoring Endpoint"""
    if not monitor:
        raise HTTPException(status_code=500, detail="Monitor not initialized")

    return {
        "latest_performance": monitor.get_latest_performance(),
        "recent_alerts": monitor.get_alert_summary(limit=10)
    }

# --- Continuous Learning Endpoints ---

@app.get("/model/health")
async def get_model_health():
    """
    Returns current active model health, metrics (C-Index, ECE, AUC),
    drift summary per feature, NPU provider, and scheduler next run times.
    """
    try:
        from continuous_learning import get_current_model_health
        from scheduler import get_scheduler_status
        health = get_current_model_health()
        health["scheduler"] = get_scheduler_status()
        health["validation_gate_thresholds"] = {
            "c_index_min": 0.88,
            "ece_max": 0.10,
            "auc_min": 0.85
        }
        return health
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model health: {e}")

@app.get("/model/drift")
async def get_model_drift():
    """
    Computes and returns real-time PSI drift analysis across all infrastructure features.
    """
    try:
        from continuous_learning import DriftDetector, DATA_STORE_PATH
        if not DATA_STORE_PATH.exists():
            raise HTTPException(status_code=404, detail="Data store not initialized")
        df = pd.read_csv(DATA_STORE_PATH)
        split_point = max(100, int(len(df) * 0.85))
        base_df = df.iloc[:split_point]
        recent_df = df.iloc[split_point:]
        if len(recent_df) < 10:
            recent_df = df.tail(100)
        detector = DriftDetector(baseline_df=base_df)
        report = detector.evaluate_drift(incoming_df=recent_df)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift evaluation failed: {e}")

@app.post("/model/retrain")
async def trigger_retrain(request: Request, user: Any = Depends(get_current_user)):
    """
    Manually triggers full stacking ensemble & RSF retraining on all available data.
    Evaluates through Validation Gate (C-Index >= 0.88, ECE <= 0.10, AUC >= 0.85).
    If promoted, hot-reloads model weights into active serving.
    """
    global system
    try:
        from continuous_learning import RetrainingOrchestrator
        orchestrator = RetrainingOrchestrator()
        result = await run_in_threadpool(orchestrator.run_retrain_cycle, "manual_api_trigger")

        # If promoted, hot-reload production model in-memory
        if result.get("promoted", False):
            try:
                system = RiskAnalysisSystem(
                    pipeline_path='pipeline.joblib',
                    ensemble_path='ensemble.joblib',
                    timeline_path='models_new/timeline.joblib' if os.path.exists('models_new/timeline.joblib') and os.path.getsize('models_new/timeline.joblib') > 1000 else 'timeline.joblib'
                )
                logging.info("[PROMOTED] Hot-reloaded promoted models into active API serving.")
            except Exception as re_err:
                logging.warning(f"Note on hot-reload: {re_err}")

        return result
    except Exception as e:
        logging.error(f"Retraining cycle failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")

class IngestRequest(BaseModel):
    records: List[Dict[str, Any]]

@app.post("/model/ingest")
async def ingest_records(request: Request, payload: IngestRequest, user: Any = Depends(get_current_user)):
    """
    Ingests new project records, validates schema, applies preprocessing,
    and appends to training data store.
    """
    try:
        from continuous_learning import ingest_project_records
        res = ingest_project_records(payload.records)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ingestion error: {e}")



# Serve static assets from dashboard directory (e.g. geojson, images)
@app.get("/{file_path:path}")
async def serve_dashboard_file(file_path: str):
    full_path = os.path.join("dashboard", file_path)
    if os.path.isfile(full_path):
        return FileResponse(full_path)
    if os.path.isfile(full_path + ".html"):
        return FileResponse(full_path + ".html")
    raise HTTPException(status_code=404, detail="File Not Found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
