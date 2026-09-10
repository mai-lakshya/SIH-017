"""
Remoteness / Urban-Tier Accessibility Delay Module
A standalone, rule-based geospatial feature-engineering and explainability engine
for land acquisition delay prediction.
"""

from .remoteness_score import evaluate_remoteness, RemotenessEvaluator
from .settlement_db import SettlementDatabase
from .geocode import GeocodingResolver
from .nearest_settlement import NearestSettlementFinder

__all__ = [
    "evaluate_remoteness",
    "RemotenessEvaluator",
    "SettlementDatabase",
    "GeocodingResolver",
    "NearestSettlementFinder",
]
