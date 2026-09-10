"""
Unit tests verifying Haversine geodesic distance calculations against known benchmark pairs.
"""

import math
import pytest
import numpy as np
from remoteness.settlement_db import SettlementDatabase, EARTH_RADIUS_KM


def haversine_formula(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def test_zero_distance_identical_point():
    d = haversine_formula(19.0760, 72.8777, 19.0760, 72.8777)
    assert round(d, 4) == 0.0


def test_mumbai_to_pune_benchmark():
    # Mumbai (18.9220, 72.8347) to Pune (18.5204, 73.8567) is ~118-122 km geodesic
    d = haversine_formula(18.9220, 72.8347, 18.5204, 73.8567)
    assert 115.0 <= d <= 125.0


def test_delhi_to_noida_benchmark():
    # Central Delhi (28.6139, 77.2090) to Noida (28.5355, 77.3910) is ~19-22 km geodesic
    d = haversine_formula(28.6139, 77.2090, 28.5355, 77.3910)
    assert 17.0 <= d <= 25.0


def test_balltree_matches_direct_haversine():
    db = SettlementDatabase()
    test_lat, test_lon = 26.9124, 75.7873  # Jaipur
    top_3 = db.query_nearest(test_lat, test_lon, k=3)

    assert len(top_3) == 3
    # Nearest to Jaipur center should be Jaipur with ~0 km distance
    nearest = top_3[0]
    assert "Jaipur" in nearest["name"]
    assert nearest["distance_km"] < 1.0

    # Cross-verify calculated distance with direct formula
    direct_d = haversine_formula(test_lat, test_lon, nearest["lat"], nearest["lon"])
    assert abs(nearest["distance_km"] - direct_d) < 0.1
