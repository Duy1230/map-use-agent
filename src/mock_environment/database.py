"""Read-only scenario database for querying the synthetic world."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from shapely.geometry import LineString, Point, Polygon

from src.world_generator.geometry_utils import haversine_m

TZ_VN = timezone(timedelta(hours=7))

TIME_RANGE_MINUTES = {
    "last_5_minutes": 5,
    "last_10_minutes": 10,
    "last_30_minutes": 30,
    "last_1_hour": 60,
    "today": 1440,
}


class ObjectNotFoundError(Exception):
    pass


class TypeMismatchError(Exception):
    pass


class ScenarioDatabase:
    """Loads a scenario dict and provides deterministic query methods."""

    def __init__(self, scenario: dict) -> None:
        self.scenario = scenario
        self.time = scenario["time"]

        self._objects: dict[str, dict] = {}
        self._by_type: dict[str, list[dict]] = {}
        for category in ("vehicles", "cameras", "polygons", "lines"):
            for obj in scenario.get("objects", {}).get(category, []):
                self._objects[obj["id"]] = obj
                self._by_type.setdefault(obj["type"], []).append(obj)

        self._time_series: dict[str, dict] = scenario.get("time_series", {})

    # --- lookups ----------------------------------------------------------

    def get_object(self, object_id: str) -> dict:
        if object_id not in self._objects:
            raise ObjectNotFoundError(f"Object {object_id!r} not found")
        return self._objects[object_id]

    def get_objects_by_type(self, obj_type: str) -> list[dict]:
        return self._by_type.get(obj_type, [])

    def resolve_reference(self, reference: str, expected_type: str | None = None) -> dict | None:
        ref_lower = reference.lower().strip()
        for obj in self._objects.values():
            if expected_type and obj["type"] != expected_type:
                continue
            if obj["id"].lower() == ref_lower or obj["name"].lower() == ref_lower:
                return obj
            for alias in obj.get("aliases", []):
                if alias.lower() == ref_lower:
                    return obj
        return None

    # --- trajectory -------------------------------------------------------

    def get_vehicle_trajectory(
        self, vehicle_id: str, time_range: str
    ) -> list[dict]:
        obj = self.get_object(vehicle_id)
        if obj["type"] != "vehicle":
            raise TypeMismatchError(
                f"Object {vehicle_id!r} is type {obj['type']!r}, not 'vehicle'"
            )

        ts_data = self._time_series.get(vehicle_id, {}).get("trajectory", [])
        if not ts_data:
            return []

        minutes = TIME_RANGE_MINUTES.get(time_range, 10)
        cutoff = datetime.fromisoformat(self.time) - timedelta(minutes=minutes)

        return [
            pt
            for pt in ts_data
            if datetime.fromisoformat(pt["timestamp"]) >= cutoff
        ]

    # --- spatial queries --------------------------------------------------

    def get_nearby_objects(
        self,
        center_id: str,
        object_type: str,
        radius_meters: float,
    ) -> list[dict]:
        center = self.get_object(center_id)
        clon, clat = center["geometry"]["coordinates"][:2]

        results = []
        for obj in self.get_objects_by_type(object_type):
            if obj["id"] == center_id:
                continue
            olon, olat = obj["geometry"]["coordinates"][:2]
            dist = haversine_m(clon, clat, olon, olat)
            if dist <= radius_meters:
                results.append({**obj, "_distance_m": round(dist, 1)})
        return results

    def get_objects_inside_polygon(
        self, polygon_id: str, object_type: str
    ) -> list[dict]:
        poly_obj = self.get_object(polygon_id)
        if poly_obj["type"] != "polygon":
            raise TypeMismatchError(
                f"Object {polygon_id!r} is type {poly_obj['type']!r}, not 'polygon'"
            )

        coords = poly_obj["geometry"]["coordinates"][0]
        shapely_poly = Polygon(coords)

        results = []
        for obj in self.get_objects_by_type(object_type):
            geom = obj["geometry"]
            if geom["type"] == "Point":
                pt = Point(geom["coordinates"][:2])
                if shapely_poly.contains(pt):
                    results.append(obj)
        return results

    def get_objects_crossing_line(
        self, line_id: str, object_type: str, time_range: str
    ) -> list[dict]:
        line_obj = self.get_object(line_id)
        if line_obj["type"] != "line":
            raise TypeMismatchError(
                f"Object {line_id!r} is type {line_obj['type']!r}, not 'line'"
            )

        line = LineString(line_obj["geometry"]["coordinates"])
        minutes = TIME_RANGE_MINUTES.get(time_range, 10)
        cutoff = datetime.fromisoformat(self.time) - timedelta(minutes=minutes)

        results = []
        for obj in self.get_objects_by_type(object_type):
            ts_data = self._time_series.get(obj["id"], {}).get("trajectory", [])
            filtered = [
                pt["coordinates"]
                for pt in ts_data
                if datetime.fromisoformat(pt["timestamp"]) >= cutoff
            ]
            if len(filtered) >= 2:
                traj = LineString(filtered)
                if line.intersects(traj):
                    results.append(obj)
        return results

    def get_polygon_area_sqm(self, polygon_id: str) -> float:
        poly_obj = self.get_object(polygon_id)
        if poly_obj["type"] != "polygon":
            raise TypeMismatchError(
                f"Object {polygon_id!r} is type {poly_obj['type']!r}, not 'polygon'"
            )
        coords = poly_obj["geometry"]["coordinates"][0]
        poly = Polygon(coords)
        # Rough conversion: 1 deg ~111km at equator. Use cos(lat) for lon.
        lat_mid = sum(c[1] for c in coords) / len(coords)
        import math
        deg2m_lat = 111_320
        deg2m_lon = 111_320 * math.cos(math.radians(lat_mid))
        # Scale Shapely area (in degrees^2) to m^2
        return poly.area * deg2m_lat * deg2m_lon

    def get_line_length_m(self, line_id: str) -> float:
        line_obj = self.get_object(line_id)
        if line_obj["type"] != "line":
            raise TypeMismatchError(
                f"Object {line_id!r} is type {line_obj['type']!r}, not 'line'"
            )
        coords = line_obj["geometry"]["coordinates"]
        total = 0.0
        for i in range(len(coords) - 1):
            total += haversine_m(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
        return round(total, 2)
