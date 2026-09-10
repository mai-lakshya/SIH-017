"""
Road Connectivity Classifier.
Identifies road connectivity type (National Highway, State Highway, District Road,
Kachcha Road, No Formal Road) using OSM Overpass queries with conservative PMGSY/NHAI fallbacks.
"""

import os
import json
import urllib.parse
import urllib.request
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# Standard valid road types
ROAD_TYPES = [
    "National Highway",
    "State Highway",
    "District Road",
    "Kachcha Road",
    "No Formal Road"
]


class RoadConnectivityClassifier:
    """
    Classifies road connectivity near a given project location.
    Includes offline conservative heuristic rules and online OSM Overpass integration.
    """

    def __init__(self, default_road_type: str = "District Road"):
        self.default_road_type = default_road_type

    def classify_road(
        self,
        lat: float,
        lon: float,
        project_type: Optional[str] = None,
        provided_road_type: Optional[str] = None,
        allow_online: bool = False
    ) -> Tuple[Dict[str, str], Optional[str]]:
        """
        Determines the road connectivity type for given coordinates.
        1. If explicitly provided by user, validates and returns it.
        2. If online OSM Overpass is reachable (and allowed), queries highway types in 3 km radius.
        3. Falls back to conservative PMGSY/NHAI domain heuristics based on project type / district context.
        """
        # 1. User/Data provided road type
        if provided_road_type:
            cleaned = str(provided_road_type).strip()
            for valid in ROAD_TYPES:
                if valid.lower() in cleaned.lower():
                    return {"type": valid, "source": "Project Specification / Field Data"}, None

        # 2. Check online OSM Overpass API if enabled
        should_online = allow_online or os.getenv("REMOTENESS_ALLOW_ONLINE", "0") == "1"
        if should_online:
            osm_result = self._query_osm_highways(lat, lon)
            if osm_result:
                return osm_result, None

        # 3. Conservative Heuristic Fallback (Offline / Demo mode)
        # Infrastructure projects (e.g. Highways/Railways) often begin on existing corridors
        if project_type:
            pt = str(project_type).lower()
            if "highway" in pt or "expressway" in pt:
                return {"type": "National Highway", "source": "NHAI GIS Alignment Heuristic"}, "fallback_nhai_alignment"
            elif "urban" in pt or "metro" in pt:
                return {"type": "State Highway", "source": "Urban PWD Corridor Heuristic"}, "fallback_urban_pwd"

        return {"type": self.default_road_type, "source": "PMGSY Rural Network Baseline"}, "fallback_pmgsy_conservative"

    def _query_osm_highways(self, lat: float, lon: float, radius_m: int = 3000) -> Optional[Dict[str, str]]:
        """
        Queries OpenStreetMap Overpass API for prominent roads within radius_m meters.
        Returns highest classification road found or None on timeout/offline.
        """
        try:
            overpass_query = f"""
            [out:json][timeout:2];
            way(around:{radius_m},{lat},{lon})["highway"];
            out tags;
            """
            url = "https://overpass-api.de/api/interpreter"
            data = urllib.parse.urlencode({"data": overpass_query}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"User-Agent": "LandIntel-Remoteness/1.0"}
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                elements = result.get("elements", [])
                
                highways = [e.get("tags", {}).get("highway") for e in elements if "tags" in e]
                if any(h in ["trunk", "motorway", "primary"] for h in highways):
                    return {"type": "National Highway", "source": "OSM Overpass Highway Network"}
                elif any(h in ["secondary"] for h in highways):
                    return {"type": "State Highway", "source": "OSM Overpass Highway Network"}
                elif any(h in ["tertiary", "unclassified", "residential"] for h in highways):
                    return {"type": "District Road", "source": "OSM Overpass Road Network"}
                elif any(h in ["track", "path"] for h in highways):
                    return {"type": "Kachcha Road", "source": "OSM Overpass Track Network"}
        except Exception:
            # Silently fallback to offline mode
            pass
        return None
