"""
Unit tests verifying mathematical monotonicity and physical design constraints.
Guarantees that remoteness delay behaves predictably without inversion anomalies.
"""

import pytest
from remoteness.remoteness_score import RemotenessEvaluator


@pytest.fixture
def evaluator():
    return RemotenessEvaluator()


def test_base_tier_delay_monotonicity(evaluator):
    """
    Design constraint:
    Metro base delay < Tier-2 < Tier-3 < Census Town < Village.
    """
    metro_delay = evaluator.compute_base_tier_delay("Metro")
    tier2_delay = evaluator.compute_base_tier_delay("Tier-2")
    tier3_delay = evaluator.compute_base_tier_delay("Tier-3")
    town_delay = evaluator.compute_base_tier_delay("Census Town")
    village_delay = evaluator.compute_base_tier_delay("Village")

    assert metro_delay == 0.0
    assert metro_delay < tier2_delay
    assert tier2_delay < tier3_delay
    assert tier3_delay < town_delay
    assert town_delay < village_delay


def test_distance_penalty_monotonicity_in_distance(evaluator):
    """
    For any fixed road type, distance penalty must be strictly non-decreasing as distance increases.
    """
    distances = [0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 150.0, 200.0]
    for road in ["National Highway", "State Highway", "District Road", "Kachcha Road", "No Formal Road"]:
        penalties = [evaluator.compute_distance_penalty(d, road) for d in distances]
        for i in range(len(penalties) - 1):
            assert penalties[i] <= penalties[i + 1], (
                f"Distance penalty inverted for road '{road}' between {distances[i]}km ({penalties[i]}d) "
                f"and {distances[i+1]}km ({penalties[i+1]}d)"
            )


def test_distance_penalty_monotonicity_across_road_types(evaluator):
    """
    For a fixed distance (e.g. 50 km), worse road quality must strictly increase travel penalty:
    National Highway <= State Highway <= District Road <= Kachcha Road <= No Formal Road.
    """
    test_distance = 50.0
    nh_p = evaluator.compute_distance_penalty(test_distance, "National Highway")
    sh_p = evaluator.compute_distance_penalty(test_distance, "State Highway")
    dr_p = evaluator.compute_distance_penalty(test_distance, "District Road")
    kr_p = evaluator.compute_distance_penalty(test_distance, "Kachcha Road")
    nr_p = evaluator.compute_distance_penalty(test_distance, "No Formal Road")

    assert nh_p < sh_p < dr_p < kr_p < nr_p


def test_terrain_modifier_scaling(evaluator):
    """
    Terrain multipliers must order: Plain (1.0) < Coastal (1.15) < Hilly (1.35) < Forest/Tribal (1.45).
    """
    res_plain = evaluator.evaluate(lat=26.9124, lon=75.7873, provided_terrain="plain", provided_road_type="District Road")
    res_coastal = evaluator.evaluate(lat=26.9124, lon=75.7873, provided_terrain="coastal", provided_road_type="District Road")
    res_hilly = evaluator.evaluate(lat=26.9124, lon=75.7873, provided_terrain="hilly", provided_road_type="District Road")
    res_forest = evaluator.evaluate(lat=26.9124, lon=75.7873, provided_terrain="forest_tribal", provided_road_type="District Road")

    assert res_plain["component_breakdown"]["terrain_adjustment_factor"] == 1.00
    assert res_coastal["component_breakdown"]["terrain_adjustment_factor"] == 1.15
    assert res_hilly["component_breakdown"]["terrain_adjustment_factor"] == 1.35
    assert res_forest["component_breakdown"]["terrain_adjustment_factor"] == 1.45

    assert res_plain["remoteness_delay_days"] <= res_coastal["remoteness_delay_days"]
    assert res_coastal["remoteness_delay_days"] <= res_hilly["remoteness_delay_days"]
    assert res_hilly["remoteness_delay_days"] <= res_forest["remoteness_delay_days"]


def test_normalized_score_bounded_in_unit_interval(evaluator):
    """
    Normalized score must always be clipped within [0.0, 1.0] across extreme values.
    """
    # Low-end check
    res_min = evaluator.evaluate(lat=18.9220, lon=72.8347, provided_road_type="National Highway", provided_terrain="plain")
    assert 0.0 <= res_min["remoteness_score_normalized"] <= 1.0

    # High-end extreme check (e.g. 500 km off-road tribal)
    res_extreme = evaluator.evaluate(lat=18.0000, lon=81.0000, provided_road_type="No Formal Road", provided_terrain="forest_tribal")
    assert 0.0 <= res_extreme["remoteness_score_normalized"] <= 1.0
