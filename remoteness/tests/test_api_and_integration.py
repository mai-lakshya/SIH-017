"""
Unit and Integration Tests for Remoteness API endpoints, Feature Engineering,
and Multicollinearity VIF checks.
"""

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from api import app, API_KEY
from remoteness.remoteness_score import evaluate_remoteness
from remoteness.integration import RemotenessFeatureEngineer


@pytest.fixture
def client():
    return TestClient(app)


def test_remoteness_api_central_mumbai(client):
    headers = {"X-API-Key": API_KEY}
    payload = {
        "latitude": 19.0760,
        "longitude": 72.8777,
        "road_type": "National Highway",
        "terrain_type": "plain"
    }
    resp = client.post("/remoteness/evaluate", json=payload, headers=headers)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Schema checks (§3.6)
    assert "site" in data
    assert "nearest_settlement" in data
    assert "road_connectivity" in data
    assert "terrain_type" in data
    assert "remoteness_delay_days" in data
    assert "remoteness_score_normalized" in data
    assert "component_breakdown" in data
    assert "data_quality_flags" in data

    # Value checks
    assert data["nearest_settlement"]["tier"] == "Metro"
    assert data["nearest_settlement"]["distance_km"] < 5.0
    assert data["remoteness_delay_days"] <= 5.0
    assert data["remoteness_score_normalized"] < 0.05


def test_remoteness_api_forested_tribal_bastar(client):
    headers = {"X-API-Key": API_KEY}
    payload = {
        "latitude": 19.1071,
        "longitude": 81.9535,
        "district": "Bastar",
        "state": "Chhattisgarh",
        "road_type": "Kachcha Road",
        "terrain_type": "forest_tribal"
    }
    resp = client.post("/remoteness/evaluate", json=payload, headers=headers)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["component_breakdown"]["fra_flat_penalty"] == 90.0
    assert data["component_breakdown"]["terrain_adjustment_factor"] >= 1.45
    assert data["remoteness_delay_days"] >= 90.0
    assert "forest_tribal_schedule_v_consent_required" in data["data_quality_flags"]


def test_remoteness_api_out_of_bounds_rejection(client):
    headers = {"X-API-Key": API_KEY}
    # Latitude 51.5°N is London, outside India
    payload = {
        "latitude": 51.5074,
        "longitude": -0.1278
    }
    resp = client.post("/remoteness/evaluate", json=payload, headers=headers)
    assert resp.status_code == 400
    assert "outside India territorial bounds" in resp.json()["detail"]


def test_feature_engineer_transform_and_vif():
    fe = RemotenessFeatureEngineer()
    dummy_data = {
        "project_id": [f"P{i:03d}" for i in range(10)],
        "state": ["Maharashtra", "Rajasthan", "Chhattisgarh", "Gujarat", "Karnataka"] * 2,
        "district": ["Pune", "Alwar", "Bastar", "Ahmedabad", "Bengaluru Urban"] * 2,
        "terrain_type": ["plain", "plain", "forest_tribal", "plain", "plain"] * 2,
        "land_area_hectares": [100.0, 250.0, 500.0, 150.0, 80.0] * 2,
        "estimated_cost_inr_crore": [200.0, 450.0, 800.0, 300.0, 150.0] * 2,
        "affected_families_count": [120, 340, 800, 210, 90] * 2,
        "title_dispute_rate_percent": [5.0, 12.0, 25.0, 8.0, 3.0] * 2,
        "compensation_multiplier_demand": [1.2, 1.5, 2.0, 1.3, 1.1] * 2,
        "fund_disbursement_percent": [45.0, 30.0, 10.0, 60.0, 80.0] * 2
    }
    df = pd.DataFrame(dummy_data)
    df_aug = fe.transform_dataframe(df, allow_online=False)

    for col in [
        "remoteness_score_normalized",
        "remoteness_delay_days",
        "settlement_tier",
        "road_connectivity_type",
        "terrain_type",
        "distance_to_nearest_settlement_km"
    ]:
        assert col in df_aug.columns

    # Check VIF
    vif_df = fe.compute_vif(df_aug, continuous_cols=[
        "remoteness_score_normalized",
        "distance_to_nearest_settlement_km",
        "land_area_hectares",
        "estimated_cost_inr_crore"
    ])
    assert not vif_df.empty
    assert "VIF" in vif_df.columns
