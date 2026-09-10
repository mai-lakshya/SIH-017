"""
Remoteness & Urban-Tier Accessibility Delay Score Evaluator.
Implements the composable rule-based delay formulation, theoretical max normalization,
and component breakdown per the Section 3.6 output specification.
"""

import os
import yaml
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from .geocode import GeocodingResolver
from .settlement_db import SettlementDatabase
from .nearest_settlement import NearestSettlementFinder
from .road_connectivity import RoadConnectivityClassifier
from .terrain import TerrainClassifier

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "remoteness_constants.yaml")


def load_constants(config_path: str = CONFIG_PATH) -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class RemotenessEvaluator:
    """
    Evaluates remoteness-driven delay and normalized score using transparent,
    travel-time grounded physical and statutory rules.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        settlement_db: Optional[SettlementDatabase] = None,
        geocoder: Optional[GeocodingResolver] = None
    ):
        self.config = config or load_constants()
        self.geocoder = geocoder or GeocodingResolver()
        self.settlement_db = settlement_db or SettlementDatabase()
        self.settlement_finder = NearestSettlementFinder(self.settlement_db, self.config)
        self.road_classifier = RoadConnectivityClassifier()
        self.terrain_classifier = TerrainClassifier()

        # Extract config constants
        self.avg_required_site_visits = float(self.config.get("avg_required_site_visits", 10.0))
        self.visit_batching_threshold_km = float(self.config.get("visit_batching_threshold_km", 150.0))
        self.visit_batching_factor = float(self.config.get("visit_batching_factor", 0.70))
        self.admin_friction_days_per_hour = float(self.config.get("admin_friction_days_per_travel_hour", 0.35))
        self.effective_speeds = self.config.get("effective_speeds_kmph", {
            "National Highway": 60.0,
            "State Highway": 45.0,
            "District Road": 30.0,
            "Kachcha Road": 15.0,
            "No Formal Road": 5.0,
            "default": 25.0
        })
        self.tier_distances = self.config.get("tier_assumed_office_distance_km", {
            "Metro": 0.0,
            "Tier-2": 5.0,
            "Tier-3": 12.0,
            "Census Town": 22.0,
            "Village": 35.0
        })
        self.base_tier_speed = float(self.config.get("base_tier_speed_kmph", 30.0))
        self.terrain_modifiers = self.config.get("terrain_modifiers", {
            "plain": 1.0,
            "coastal": 1.15,
            "hilly": 1.35,
            "forest_tribal": 1.45
        })
        self.fra_flat_penalty_days = float(self.config.get("fra_flat_penalty_days", 90.0))
        self.theoretical_max_delay_days = float(self.config.get("theoretical_max_delay_days", 385.0))

    def compute_base_tier_delay(self, tier: str) -> float:
        """
        Estimates the baseline administrative delay in days associated with the settlement tier.
        Reflects travel distance to the nearest tier-appropriate office:
        base_delay = (2 * assumed_distance_km / base_speed_kmph) * visits * friction
        """
        assumed_dist = float(self.tier_distances.get(tier, self.tier_distances.get("Village", 35.0)))
        if assumed_dist <= 0.0:
            return 0.0
        # Round trip hours per visit
        round_trip_hours = (2.0 * assumed_dist) / max(self.base_tier_speed, 1.0)
        # Multiplied by standard touchpoints and scheduling friction
        delay_days = round_trip_hours * self.avg_required_site_visits * self.admin_friction_days_per_hour
        return round(delay_days, 2)

    def compute_distance_penalty(self, distance_km: float, road_type: str) -> float:
        """
        Computes the additional delay days contributed by the physical distance to the settlement
        given the road connectivity type. Applies marginal visit batching beyond 150 km,
        flattening the marginal cost per additional km while preserving strict monotonicity.
        """
        dist = max(0.0, float(distance_km))
        if dist <= 0.0:
            return 0.0

        effective_speed = float(self.effective_speeds.get(road_type, self.effective_speeds.get("default", 25.0)))
        effective_speed = max(1.0, effective_speed)

        # Marginal trip batching beyond 150 km (Section 4.1):
        # Full 10 visits for the first 150 km, and consolidated 7 visits for distance beyond 150 km.
        if dist > self.visit_batching_threshold_km:
            threshold = self.visit_batching_threshold_km
            hours_base = (2.0 * threshold / effective_speed) * self.avg_required_site_visits
            excess_dist = dist - threshold
            hours_excess = (2.0 * excess_dist / effective_speed) * (self.avg_required_site_visits * self.visit_batching_factor)
            total_travel_hours = hours_base + hours_excess
        else:
            total_travel_hours = (2.0 * dist / effective_speed) * self.avg_required_site_visits

        penalty_days = total_travel_hours * self.admin_friction_days_per_hour
        return round(penalty_days, 2)

    def evaluate(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        address: Optional[str] = None,
        project_type: Optional[str] = None,
        district: Optional[str] = None,
        state: Optional[str] = None,
        provided_road_type: Optional[str] = None,
        provided_terrain: Optional[str] = None,
        road_type: Optional[str] = None,
        terrain_type: Optional[str] = None,
        allow_online: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the full remoteness evaluation pipeline and returns the Section 3.6 JSON schema.
        """
        eff_road = provided_road_type or road_type
        eff_terrain = provided_terrain or terrain_type
        eff_address = address
        if not eff_address and district and state and str(district).strip() not in ["", "Unknown", "nan"]:
            eff_address = f"{district}, {state}"

        # 1. Resolve & validate geographic coordinates
        (resolved_lat, resolved_lon), geo_flags = self.geocoder.resolve(
            lat=lat, lon=lon, address=eff_address, allow_online=allow_online
        )

        # 2. Nearest settlement lookup & urban tier classification
        primary_settlement, top_3, stl_flags = self.settlement_finder.find_nearest_settlements(
            resolved_lat, resolved_lon, k=3
        )

        # 3. Road connectivity classification
        road_conn, road_flag = self.road_classifier.classify_road(
            resolved_lat, resolved_lon, project_type=project_type, provided_road_type=eff_road, allow_online=allow_online
        )
        final_road_type = road_conn["type"]

        # 4. Terrain & Forest / Tribal classification
        eff_dist = district or primary_settlement.get("district")
        final_terrain_type, is_forest_tribal = self.terrain_classifier.classify_terrain(
            resolved_lat, resolved_lon, district=eff_dist, provided_terrain=eff_terrain
        )

        # 5. Component breakdown calculations
        base_tier_delay = self.compute_base_tier_delay(primary_settlement["tier"])
        dist_km = primary_settlement["distance_km"]
        distance_penalty = self.compute_distance_penalty(dist_km, final_road_type)

        terrain_factor = float(self.terrain_modifiers.get(final_terrain_type, 1.0))
        fra_penalty = self.fra_flat_penalty_days if is_forest_tribal else 0.0

        # Formula: (base_delay + distance_penalty) * terrain_modifier + fra_flat_penalty
        subtotal = (base_tier_delay + distance_penalty) * terrain_factor
        total_delay_days = round(subtotal + fra_penalty, 2)

        # 6. Normalized Remoteness Score (0 - 1)
        normalized_score = float(np.clip(total_delay_days / self.theoretical_max_delay_days, 0.0, 1.0))
        normalized_score = round(normalized_score, 4)

        # 7. Collect data quality flags
        all_flags = list(set(geo_flags + stl_flags + ([road_flag] if road_flag else [])))
        if is_forest_tribal:
            all_flags.append("forest_tribal_schedule_v_consent_required")

        return {
            "site": {
                "lat": resolved_lat,
                "lon": resolved_lon
            },
            "nearest_settlement": primary_settlement,
            "top_3_settlements": [
                {
                    "name": s["name"],
                    "tier": s["tier"],
                    "population": int(s["population"]),
                    "distance_km": float(s["distance_km"])
                }
                for s in top_3
            ],
            "road_connectivity": road_conn,
            "terrain_type": final_terrain_type,
            "remoteness_delay_days": total_delay_days,
            "remoteness_score_normalized": normalized_score,
            "component_breakdown": {
                "base_tier_delay": base_tier_delay,
                "distance_penalty": distance_penalty,
                "terrain_adjustment_factor": terrain_factor,
                "fra_flat_penalty": fra_penalty
            },
            "data_quality_flags": all_flags
        }


# Global evaluator singleton instance
_GLOBAL_EVALUATOR: Optional[RemotenessEvaluator] = None


def evaluate_remoteness(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    address: Optional[str] = None,
    project_type: Optional[str] = None,
    district: Optional[str] = None,
    state: Optional[str] = None,
    road_type: Optional[str] = None,
    terrain_type: Optional[str] = None,
    provided_road_type: Optional[str] = None,
    provided_terrain: Optional[str] = None,
    allow_online: bool = False
) -> Dict[str, Any]:
    """Convenience helper function to evaluate remoteness using the singleton evaluator."""
    global _GLOBAL_EVALUATOR
    if _GLOBAL_EVALUATOR is None:
        _GLOBAL_EVALUATOR = RemotenessEvaluator()
    return _GLOBAL_EVALUATOR.evaluate(
        lat=lat,
        lon=lon,
        address=address,
        project_type=project_type,
        district=district,
        state=state,
        provided_road_type=provided_road_type or road_type,
        provided_terrain=provided_terrain or terrain_type,
        allow_online=allow_online
    )
