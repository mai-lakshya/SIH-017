"""
Section 10 Benchmark Tests: Snapshot Tests on Known Geographic Locations.
Directional validation confirming that pipeline outputs align with physical and institutional reality.
"""

import pytest
from remoteness.remoteness_score import evaluate_remoteness


def test_site_in_central_mumbai():
    """
    Site: Central Mumbai (Fort / Nariman Point area)
    Expected Tier: Metro
    Expected Distance: < 5 km
    Expected Direction: Low delay (0 - 10 days)
    """
    res = evaluate_remoteness(
        lat=18.9220,
        lon=72.8347,
        road_type="National Highway",
        terrain_type="plain"
    )

    nearest = res["nearest_settlement"]
    assert nearest["tier"] == "Metro"
    assert nearest["distance_km"] < 5.0
    assert 0.0 <= res["remoteness_delay_days"] <= 10.0
    assert res["remoteness_score_normalized"] < 0.05
    assert res["component_breakdown"]["fra_flat_penalty"] == 0.0


def test_site_adjacent_to_tier3_city():
    """
    Site: Alwar outskirts (~3 km from Alwar city center, Rajasthan)
    Expected Tier: Tier-3
    Expected Distance: < 5 km
    Expected Direction: Low-moderate delay (~3 - 15 days)
    """
    # Alwar center is ~27.5530, 76.6346; test a point 3 km to the east (27.5530, 76.6650)
    res = evaluate_remoteness(
        lat=27.5530,
        lon=76.6650,
        district="Alwar",
        road_type="District Road",
        terrain_type="plain"
    )

    nearest = res["nearest_settlement"]
    assert "Alwar" in nearest["name"]
    assert nearest["tier"] in ["Tier-3", "Tier-2"]
    assert nearest["distance_km"] < 5.0
    assert 2.0 <= res["remoteness_delay_days"] <= 15.0
    assert res["component_breakdown"]["fra_flat_penalty"] == 0.0


def test_site_in_forested_tribal_district():
    """
    Site: Interior Bastar forest tract (Chhattisgarh) ~70-80 km from nearest town
    Expected Tier: Village or Town
    Expected Direction: High delay (>130 days) with explicit FRA/tribal-consent flag (90 days)
    """
    # Location deep in Bastar forest corridor (19.5°N, 80.8°E) ~73 km from Kanker/Bastar
    res = evaluate_remoteness(
        lat=19.5000,
        lon=80.8000,
        district="Bastar",
        road_type="Kachcha Road",
        terrain_type="forest_tribal"
    )

    assert res["terrain_type"] == "forest_tribal"
    assert res["nearest_settlement"]["distance_km"] >= 60.0
    assert res["component_breakdown"]["fra_flat_penalty"] == 90.0
    assert res["component_breakdown"]["terrain_adjustment_factor"] == 1.45
    # Total delay is high (>130 days) due to distance + kachcha road + terrain multiplier + FRA penalty
    assert res["remoteness_delay_days"] >= 130.0
    assert res["remoteness_score_normalized"] >= 0.35


def test_site_remote_kachcha_road_150km():
    """
    Site: Ultra-remote project site ~150 km from major settlements with kachcha road access
    Expected Direction: Very high delay (>100 days), dominated by distance penalty + road modifier
    """
    # Remote high-altitude plateau in eastern Ladakh (~168 km from nearest district town)
    res = evaluate_remoteness(
        lat=33.0000,
        lon=79.0000,
        district="Leh",
        road_type="Kachcha Road",
        terrain_type="hilly"
    )

    assert res["nearest_settlement"]["distance_km"] >= 140.0
    dist_penalty = res["component_breakdown"]["distance_penalty"]
    # Distance penalty for ~150+ km at 15 km/h with 10 visits * 0.35 friction is >= 70 days
    assert dist_penalty >= 70.0
    assert res["remoteness_delay_days"] >= 100.0
