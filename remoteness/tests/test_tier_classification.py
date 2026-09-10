"""
Unit tests for Census 2011 population-based urban tier classification and agglomeration logic.
"""

import pytest
from remoteness.settlement_db import SettlementDatabase
from remoteness.nearest_settlement import NearestSettlementFinder


def test_population_tier_thresholds():
    db = SettlementDatabase()
    assert db._classify_tier(2500000) == "Metro"
    assert db._classify_tier(1000000) == "Metro"
    assert db._classify_tier(999999) == "Tier-2"
    assert db._classify_tier(500000) == "Tier-2"
    assert db._classify_tier(499999) == "Tier-3"
    assert db._classify_tier(100000) == "Tier-3"
    assert db._classify_tier(99999) == "Census Town"
    assert db._classify_tier(5000) == "Census Town"
    assert db._classify_tier(4999) == "Village"
    assert db._classify_tier(500) == "Village"


def test_urban_agglomeration_satellite_override():
    """
    If a satellite settlement / village is within agglomeration radius (<15km)
    of a Metro, the finder adopts the Metro's institutional tier for base delay.
    """
    finder = NearestSettlementFinder()
    # Coordinates in Navi Mumbai / Thane belt (near Mumbai)
    primary, candidates, flags = finder.find_nearest_settlements(19.0400, 72.9500, k=3)

    tiers_in_top3 = [c["tier"] for c in candidates]
    assert "Metro" in tiers_in_top3 or primary["tier"] == "Metro"
    assert primary["distance_km"] < 15.0
    assert primary["population_data_vintage"] == "Census 2011"
