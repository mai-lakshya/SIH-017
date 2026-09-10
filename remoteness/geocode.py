"""
Geospatial Input Handling & Geocoding Resolver.
Validates Indian coordinates, provides rate-limited/cached geocoding,
and supports offline fallback for resilient demo execution.
"""

import os
import re
import json
import sqlite3
import urllib.parse
import urllib.request
import logging
from typing import Tuple, Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Bounding box of India territory (including territorial waters and islands)
DEFAULT_INDIA_BBOX = {
    "min_lat": 6.0,
    "max_lat": 38.0,
    "min_lon": 68.0,
    "max_lon": 98.0
}

CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), "cache", "geocache.db")


class GeocodingResolver:
    """
    Handles resolution of geographic inputs to validated (lat, lon) coordinates
    with persistent SQLite caching, OSM Nominatim lookup, Google Maps API fallback,
    and offline dictionary lookups.
    """

    def __init__(self, cache_db_path: str = CACHE_DB_PATH, bbox: dict = None):
        self.bbox = bbox or DEFAULT_INDIA_BBOX
        self.cache_db_path = cache_db_path
        self._init_cache_db()
        self._offline_districts = self._load_offline_districts()

    def _init_cache_db(self):
        """Initializes local SQLite cache to avoid rate-limiting and enable offline recall."""
        os.makedirs(os.path.dirname(self.cache_db_path), exist_ok=True)
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS geocache (
                        query_key TEXT PRIMARY KEY,
                        lat REAL,
                        lon REAL,
                        display_name TEXT,
                        provider TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not initialize geocache SQLite: {e}")

    def _load_offline_districts(self) -> Dict[str, Tuple[float, float]]:
        """Loads offline district coordinates dictionary if available."""
        districts = {}
        for candidate_path in [
            os.path.join(os.path.dirname(__file__), "..", "district_coordinates.json"),
            "district_coordinates.json",
        ]:
            candidate_path = os.path.abspath(candidate_path)
            if os.path.exists(candidate_path):
                try:
                    with open(candidate_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                        for k, v in raw.items():
                            coords = (float(v[0]), float(v[1]))
                            k_clean = k.strip().lower()
                            districts[k_clean] = coords
                            if "|" in k:
                                state, dist = k.split("|", 1)
                                s_clean = state.strip().lower()
                                d_clean = dist.strip().lower()
                                districts[d_clean] = coords
                                districts[f"{d_clean}, {s_clean}"] = coords
                                districts[f"{s_clean}|{d_clean}"] = coords
                                districts[f"{d_clean} {s_clean}"] = coords
                                districts[f"{d_clean}, india"] = coords
                                districts[f"{d_clean}, {s_clean}, india"] = coords
                    break
                except Exception as e:
                    logger.warning(f"Failed to load district coordinates from {candidate_path}: {e}")
        return districts

    def validate_india_coordinates(self, lat: float, lon: float) -> Tuple[bool, Optional[str]]:
        """
        Validates if coordinates fall within the geographical boundary of India.
        """
        if lat is None or lon is None:
            return False, "Null coordinates supplied."
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            return False, f"Non-numeric coordinates supplied: lat={lat}, lon={lon}"

        if not (self.bbox["min_lat"] <= lat <= self.bbox["max_lat"]):
            return False, (
                f"Latitude {lat:.4f}° is outside India territorial bounds "
                f"[{self.bbox['min_lat']}°N, {self.bbox['max_lat']}°N]."
            )
        if not (self.bbox["min_lon"] <= lon <= self.bbox["max_lon"]):
            return False, (
                f"Longitude {lon:.4f}° is outside India territorial bounds "
                f"[{self.bbox['min_lon']}°E, {self.bbox['max_lon']}°E]."
            )
        return True, None

    def _get_cached(self, query_key: str) -> Optional[Tuple[float, float, str, str]]:
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT lat, lon, display_name, provider FROM geocache WHERE query_key = ?",
                    (query_key.strip().lower(),)
                )
                row = cursor.fetchone()
                if row:
                    return float(row[0]), float(row[1]), str(row[2]), str(row[3])
        except Exception:
            pass
        return None

    def _save_cache(self, query_key: str, lat: float, lon: float, display_name: str, provider: str):
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO geocache (query_key, lat, lon, display_name, provider) VALUES (?, ?, ?, ?, ?)",
                    (query_key.strip().lower(), lat, lon, display_name, provider)
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to cache geocode result: {e}")

    def geocode_address(
        self,
        address: str,
        google_api_key: Optional[str] = None,
        allow_online: bool = False
    ) -> Tuple[Optional[Tuple[float, float]], List[str], Optional[str]]:
        """
        Resolves an address / village name to coordinates.
        Order of resolution:
        1. Local SQLite cache
        2. Offline District / Town lookup
        3. OpenStreetMap Nominatim API (if allow_online=True)
        4. Optional Google Maps API (if allow_online=True and key provided)
        """
        clean_addr = address.strip()
        flags: List[str] = ["geocoded_from_address"]
        norm_key = clean_addr.lower()

        # 1. Check local cache
        cached = self._get_cached(norm_key)
        if cached:
            lat, lon, name, prov = cached
            is_valid, _ = self.validate_india_coordinates(lat, lon)
            if is_valid:
                flags.append(f"cache_hit_{prov}")
                return (lat, lon), flags, name

        # 2. Check offline district match
        # Match 'State|District' or just 'District'
        if norm_key in self._offline_districts:
            lat, lon = self._offline_districts[norm_key]
            flags.append("offline_district_match")
            self._save_cache(norm_key, lat, lon, clean_addr, "offline_districts")
            return (lat, lon), flags, clean_addr

        # Partial match for district
        for dist_name, coords in self._offline_districts.items():
            if dist_name in norm_key or norm_key in dist_name:
                lat, lon = coords
                flags.append("offline_district_partial_match")
                self._save_cache(norm_key, lat, lon, f"{dist_name}, India", "offline_districts")
                return (lat, lon), flags, dist_name

        # 3. OSM Nominatim (Rate-limited, online) - only if allow_online is enabled
        should_online = allow_online or os.getenv("REMOTENESS_ALLOW_ONLINE", "0") == "1"
        if should_online:
            try:
                query = urllib.parse.quote(f"{clean_addr}, India")
                url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1&countrycodes=in"
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "LandIntel-RemotenessModule/1.0 (sih-land-acquisition-analytics)"}
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data and len(data) > 0:
                        lat = float(data[0]["lat"])
                        lon = float(data[0]["lon"])
                        display = data[0].get("display_name", clean_addr)
                        is_valid, err = self.validate_india_coordinates(lat, lon)
                        if is_valid:
                            flags.append("osm_nominatim_resolved")
                            self._save_cache(norm_key, lat, lon, display, "osm_nominatim")
                            return (lat, lon), flags, display
            except Exception as e:
                logger.info(f"OSM Nominatim resolution skipped/failed for '{clean_addr}': {e}")

            # 4. Google Maps Geocoding fallback if key provided
            if google_api_key or os.getenv("GOOGLE_MAPS_API_KEY"):
                key = google_api_key or os.getenv("GOOGLE_MAPS_API_KEY")
                try:
                    g_query = urllib.parse.quote(f"{clean_addr}, India")
                    g_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={g_query}&key={key}"
                    req = urllib.request.Request(g_url)
                    with urllib.request.urlopen(req, timeout=3.0) as resp:
                        g_data = json.loads(resp.read().decode("utf-8"))
                        if g_data.get("status") == "OK" and g_data.get("results"):
                            loc = g_data["results"][0]["geometry"]["location"]
                            lat = float(loc["lat"])
                            lon = float(loc["lng"])
                            display = g_data["results"][0].get("formatted_address", clean_addr)
                            is_valid, err = self.validate_india_coordinates(lat, lon)
                            if is_valid:
                                flags.append("google_geocoding_resolved")
                                self._save_cache(norm_key, lat, lon, display, "google_maps")
                                return (lat, lon), flags, display
                except Exception as e:
                    logger.info(f"Google Maps geocoding failed: {e}")

        # Fallback to central India if completely unresolved
        flags.append("geocoding_failed_fallback_default")
        return None, flags, None

    def resolve(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        address: Optional[str] = None,
        allow_online: bool = False
    ) -> Tuple[Tuple[float, float], List[str]]:
        """
        Resolves input coordinates or address to validated (lat, lon) within India.
        Raises ValueError if out of bounds or unresolvable.
        """
        flags: List[str] = []

        if lat is not None and lon is not None:
            is_valid, err = self.validate_india_coordinates(lat, lon)
            if not is_valid:
                raise ValueError(err)
            return (round(float(lat), 6), round(float(lon), 6)), flags

        if address:
            coords, addr_flags, _ = self.geocode_address(address, allow_online=allow_online)
            flags.extend(addr_flags)
            if coords:
                return (round(coords[0], 6), round(coords[1], 6)), flags
            raise ValueError(f"Could not resolve address '{address}' to valid coordinates within India.")

        raise ValueError("Must provide either (lat, lon) coordinates or an address string.")

