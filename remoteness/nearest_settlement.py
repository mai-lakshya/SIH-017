"""
Nearest Settlement Finder & Urban-Tier Agglomeration Classifier.
Resolves nearest settlements, classifies urban tiers, and handles the
urban agglomeration edge case (e.g., satellite towns adjacent to metros).
"""

import os
import yaml
from typing import List, Dict, Any, Tuple
from .settlement_db import SettlementDatabase

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "remoteness_constants.yaml")

# Hierarchy rank of tiers for agglomeration comparison (higher rank = larger urban hierarchy)
TIER_RANKS = {
    "Village": 1,
    "Census Town": 2,
    "Tier-3": 3,
    "Tier-2": 4,
    "Metro": 5
}


def load_config(config_path: str = CONFIG_PATH) -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class NearestSettlementFinder:
    """
    Identifies top nearest settlements from the spatial database,
    evaluates urban agglomeration effects, and returns tier classifications.
    """

    def __init__(self, settlement_db: SettlementDatabase = None, config: dict = None):
        self.db = settlement_db or SettlementDatabase()
        self.config = config or load_config()
        self.agglomeration_radius_km = float(self.config.get("agglomeration_radius_km", 15.0))

    def find_nearest_settlements(self, lat: float, lon: float, k: int = 3) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
        """
        Queries top-k nearest settlements for given coordinates.
        Handles the urban agglomeration edge case:
        If a higher-tier settlement is within `agglomeration_radius_km` in top-k,
        prefer the higher tier for base administrative services,
        while maintaining the literal nearest distance for distance penalty.

        Returns:
            (primary_settlement_metadata, top_k_settlements, data_quality_flags)
        """
        flags: List[str] = []
        candidates = self.db.query_nearest(lat, lon, k=k)

        if not candidates:
            flags.append("no_settlement_match")
            # Fallback conservative village representation
            fallback = {
                "name": "Unidentified Rural Locality",
                "tier": "Village",
                "population": 1500,
                "distance_km": 50.0,
                "population_data_vintage": "Census 2011 (Extrapolated)"
            }
            return fallback, [fallback], flags

        nearest = candidates[0]
        literal_distance = nearest["distance_km"]
        effective_tier = nearest["tier"]
        effective_name = nearest["name"]
        effective_pop = nearest["population"]

        # Check for Urban Agglomeration edge case across top-3
        # If a materially higher-tier city is within agglomeration_radius_km (e.g. 15 km),
        # staff and resources flow from that city, so base administrative tier is adopted.
        nearest_rank = TIER_RANKS.get(effective_tier, 1)
        for other in candidates[1:]:
            other_dist = other["distance_km"]
            other_tier = other["tier"]
            other_rank = TIER_RANKS.get(other_tier, 1)

            if other_dist <= self.agglomeration_radius_km and other_rank > nearest_rank:
                flags.append(f"urban_agglomeration_adopted_{other_tier.lower()}_from_{other['name']}")
                effective_tier = other_tier
                effective_name = f"{nearest['name']} (Agglomeration with {other['name']})"
                effective_pop = other["population"]
                nearest_rank = other_rank

        primary = {
            "name": effective_name,
            "tier": effective_tier,
            "population": int(effective_pop),
            "distance_km": float(literal_distance),
            "population_data_vintage": "Census 2011"
        }

        # Flag if distance is unusually large (>100 km)
        if literal_distance > 100.0:
            flags.append("high_distance_outlier")

        return primary, candidates, flags
