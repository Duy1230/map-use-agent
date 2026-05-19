"""Read-only scenario database for aircraft track analysis queries."""

from __future__ import annotations

import math
import statistics
from datetime import timedelta, timezone
from typing import Any

from src.domain.profile import DomainProfile
from src.domain.scenario import iter_scenario_objects
from src.world_generator.geometry_utils import haversine_m

TZ_VN = timezone(timedelta(hours=7))


class TrackNotFoundError(Exception):
    pass


class POINotFoundError(Exception):
    pass


class AircraftScenarioDatabase:
    """Loads an aircraft scenario and provides deterministic query methods."""

    def __init__(self, scenario: dict, *, profile: DomainProfile | None = None) -> None:
        self.scenario = scenario
        self.profile = profile
        self.time = scenario.get("time", "")

        self._objects: dict[str, dict] = {}
        self._by_type: dict[str, list[dict]] = {}
        for obj in iter_scenario_objects(scenario, profile):
            self._objects[obj["id"]] = obj
            self._by_type.setdefault(obj.get("type", ""), []).append(obj)

        self._time_series: dict[str, dict] = scenario.get("time_series", {})
        self._flight_plans: dict[str, dict] = scenario.get("flight_plans", {})
        self._aircraft_models: dict[str, str] = scenario.get("aircraft_models", {})
        self._historical_tracks: list[dict] = scenario.get("historical_tracks", [])

    def _get_track(self, track_id: str) -> dict:
        obj = self._objects.get(track_id)
        if obj is None or obj.get("type") != "track":
            raise TrackNotFoundError(f"Track {track_id!r} not found")
        return obj

    def _get_trajectory(self, track_id: str) -> list[dict]:
        return self._time_series.get(track_id, {}).get("trajectory", [])

    # ------------------------------------------------------------------
    # Tool 1: getTrackInfo
    # ------------------------------------------------------------------
    def get_track_info(self, track_id: str, fields: list[str]) -> dict:
        track = self._get_track(track_id)
        traj = self._get_trajectory(track_id)
        result: dict[str, Any] = {"trackId": track_id}

        want_all = "all" in fields
        if want_all or "identity" in fields:
            result["identity"] = {
                "callSign": track.get("callSign", ""),
                "registration": track.get("registration", ""),
                "type": track.get("targetType", ""),
                "aircraftModel": track.get("aircraftModel", ""),
            }
        if want_all or "timespan" in fields:
            if traj:
                result["timespan"] = {
                    "start": traj[0]["timestamp"],
                    "end": traj[-1]["timestamp"],
                    "durationSec": len(traj) * 10,
                }
            else:
                result["timespan"] = {}
        if want_all or "startPoint" in fields:
            if traj:
                p = traj[0]
                result["startPoint"] = _point_to_geo(p)
            else:
                result["startPoint"] = None
        if want_all or "endPoint" in fields:
            if traj:
                p = traj[-1]
                result["endPoint"] = _point_to_geo(p)
            else:
                result["endPoint"] = None
        if want_all or "latestPoint" in fields:
            if traj:
                p = traj[-1]
                result["latestPoint"] = _point_to_geo(p)
            else:
                result["latestPoint"] = None
        if want_all or "signalGaps" in fields:
            gaps = track.get("properties", {}).get("signalGaps", [])
            result["signalGaps"] = [
                {
                    "description": f"Signal gap {g['durationSec']}s",
                    "severity": "warn",
                    "timeRange": {"start": g["start"], "end": g["end"], "durationSec": g["durationSec"]},
                    "geoFeatures": [
                        {"type": "point", "geometry": g.get("startPoint", {}), "properties": {"label": "Gap start"}},
                        {"type": "point", "geometry": g.get("endPoint", {}), "properties": {"label": "Gap end"}},
                    ],
                }
                for g in gaps
            ]
        return result

    # ------------------------------------------------------------------
    # Tool 2: getTrackPoints
    # ------------------------------------------------------------------
    def get_track_points(
        self, track_id: str, time_range: dict | None = None, sampling: str = "downsample-10s"
    ) -> list[dict]:
        self._get_track(track_id)
        traj = self._get_trajectory(track_id)
        if time_range:
            start = time_range.get("start", "")
            end = time_range.get("end", "")
            traj = [p for p in traj if start <= p["timestamp"] <= end] if start and end else traj

        if sampling == "downsample-10s":
            return traj
        if sampling == "downsample-1s":
            return traj
        return traj

    # ------------------------------------------------------------------
    # Tool 3: getKinematicStats
    # ------------------------------------------------------------------
    def get_kinematic_stats(self, track_id: str, dimension: str) -> dict:
        self._get_track(track_id)
        traj = self._get_trajectory(track_id)
        if not traj:
            return {"mean": 0, "min": 0, "max": 0}

        values = [p.get(dimension, p.get("speed", 0)) for p in traj]
        if not values:
            return {"mean": 0, "min": 0, "max": 0}

        mean_v = statistics.mean(values)
        min_v = min(values)
        max_v = max(values)
        min_idx = values.index(min_v)
        max_idx = values.index(max_v)

        num_bins = 10
        bin_width = (max_v - min_v) / num_bins if max_v > min_v else 1
        bins = [round(min_v + i * bin_width, 2) for i in range(num_bins + 1)]
        counts = [0] * num_bins
        for v in values:
            idx = min(int((v - min_v) / bin_width), num_bins - 1) if bin_width > 0 else 0
            counts[idx] += 1

        min_pt = traj[min_idx]
        max_pt = traj[max_idx]

        return {
            "mean": round(mean_v, 2),
            "min": round(min_v, 2),
            "max": round(max_v, 2),
            "atMinTime": min_pt["timestamp"],
            "atMaxTime": max_pt["timestamp"],
            "distribution": {"bins": bins, "counts": counts},
            "series": [{"t": p["timestamp"], "v": round(p.get(dimension, p.get("speed", 0)), 2)} for p in traj[::max(len(traj) // 100, 1)]],
            "geoFeatures": [
                {"type": "point", "geometry": {"lat": min_pt["lat"], "lon": min_pt["lon"]}, "properties": {"label": f"Min {dimension}"}},
                {"type": "point", "geometry": {"lat": max_pt["lat"], "lon": max_pt["lon"]}, "properties": {"label": f"Max {dimension}"}},
            ],
        }

    # ------------------------------------------------------------------
    # Tool 4: analyzeSpatial
    # ------------------------------------------------------------------
    def analyze_spatial(self, track_id: str, aspect: str) -> list[dict]:
        self._get_track(track_id)
        traj = self._get_trajectory(track_id)
        if not traj:
            return []

        if aspect == "overflownAreas":
            return self._analyze_overflown(traj)
        if aspect == "restrictedZones":
            return self._analyze_restricted(track_id, traj)
        if aspect == "commonRoutes":
            return self._analyze_common_routes(traj)
        if aspect == "notablePoints":
            return self._analyze_notable(traj)
        if aspect == "primaryHeading":
            return self._analyze_heading(traj)
        return []

    def _analyze_overflown(self, traj: list[dict]) -> list[dict]:
        zones = self._by_type.get("restricted_zone", [])
        findings: list[dict] = []
        for zone in zones:
            coords = zone["geometry"]["coordinates"][0]
            from shapely.geometry import Point, Polygon
            poly = Polygon(coords)
            for p in traj[::10]:
                if poly.contains(Point(p["lon"], p["lat"])):
                    findings.append({
                        "description": f"Bay qua khu vực {zone['name']}",
                        "severity": "info",
                        "timeRange": {"start": p["timestamp"], "end": p["timestamp"]},
                        "geoFeatures": [{"type": "polygon", "geometry": {"coordinates": zone["geometry"]["coordinates"]}, "properties": {"label": zone["name"]}}],
                    })
                    break
        return findings

    def _analyze_restricted(self, track_id: str, traj: list[dict]) -> list[dict]:
        zones = self._by_type.get("restricted_zone", [])
        findings: list[dict] = []
        for zone in zones:
            coords = zone["geometry"]["coordinates"][0]
            from shapely.geometry import Point, Polygon
            poly = Polygon(coords)
            intrusion_pts = [p for p in traj if poly.contains(Point(p["lon"], p["lat"]))]
            if intrusion_pts:
                findings.append({
                    "description": f"Xâm phạm {zone['name']}",
                    "severity": "critical",
                    "timeRange": {"start": intrusion_pts[0]["timestamp"], "end": intrusion_pts[-1]["timestamp"], "durationSec": len(intrusion_pts) * 10},
                    "geoFeatures": [
                        {"type": "line", "geometry": {"coordinates": [[p["lon"], p["lat"]] for p in intrusion_pts]}, "properties": {"label": "Intrusion segment"}},
                        {"type": "polygon", "geometry": {"coordinates": zone["geometry"]["coordinates"]}, "properties": {"label": zone["name"]}},
                    ],
                })
        return findings

    def _analyze_common_routes(self, traj: list[dict]) -> list[dict]:
        routes = self._by_type.get("common_route", [])
        findings: list[dict] = []
        for route in routes:
            route_coords = route["geometry"]["coordinates"]
            if len(route_coords) < 2 or len(traj) < 2:
                continue
            mid_route = route_coords[len(route_coords) // 2]
            mid_traj = traj[len(traj) // 2]
            dist = haversine_m(mid_route[0], mid_route[1], mid_traj["lon"], mid_traj["lat"])
            if dist < 50_000:
                findings.append({
                    "description": f"Có thể theo tuyến {route['name']}",
                    "severity": "info",
                    "geoFeatures": [{"type": "line", "geometry": route["geometry"], "properties": {"label": route["name"]}}],
                    "metrics": {"matchDistanceKm": round(dist / 1000, 1)},
                })
        return findings

    def _analyze_notable(self, traj: list[dict]) -> list[dict]:
        if len(traj) < 10:
            return []
        speeds = [p.get("speed", 0) for p in traj]
        mean_speed = statistics.mean(speeds)
        findings: list[dict] = []
        for i, p in enumerate(traj):
            if abs(p.get("speed", 0) - mean_speed) > mean_speed * 0.5:
                findings.append({
                    "description": f"Tốc độ bất thường tại {p['timestamp']}",
                    "severity": "warn",
                    "timeRange": {"start": p["timestamp"], "end": p["timestamp"]},
                    "geoFeatures": [_point_to_geofeature(p, "Notable point")],
                })
                if len(findings) >= 5:
                    break
        return findings

    def _analyze_heading(self, traj: list[dict]) -> list[dict]:
        headings = [p.get("heading", 0) for p in traj]
        if not headings:
            return []
        primary = statistics.mode([int(h / 45) * 45 for h in headings])
        return [{
            "description": f"Hướng bay chính: {primary}°",
            "severity": "info",
            "metrics": {"primaryHeading": primary},
            "geoFeatures": [],
        }]

    # ------------------------------------------------------------------
    # Tool 5: detectBehavior
    # ------------------------------------------------------------------
    def detect_behavior(self, track_id: str, pattern: str, options: dict | None = None) -> list[dict]:
        track = self._get_track(track_id)
        traj = self._get_trajectory(track_id)
        if not traj:
            return []

        tags = track.get("properties", {}).get("behaviorTags", [])
        pattern_to_tag = {
            "abnormalAcceleration": "speed_anomaly",
            "speedThresholdExceeded": "speed_anomaly",
            "hovering": "hovering",
            "abnormalAltitudeChange": "altitude_anomaly",
            "sustainedLowAltitude": "sustained_low",
            "suddenDirectionChange": "uturn",
            "circling": "circling",
            "zigzag": "zigzag",
            "uTurn": "uturn",
        }
        tag = pattern_to_tag.get(pattern, pattern)
        if tag not in tags:
            return []

        mid = len(traj) // 2
        mid_pt = traj[mid]
        return [{
            "description": f"Phát hiện {pattern} tại {mid_pt['timestamp']}",
            "severity": "warn",
            "timeRange": {"start": traj[max(mid - 20, 0)]["timestamp"], "end": traj[min(mid + 20, len(traj) - 1)]["timestamp"]},
            "geoFeatures": [_point_to_geofeature(mid_pt, pattern)],
            "metrics": {},
        }]

    # ------------------------------------------------------------------
    # Tool 6: lookupAircraftModel
    # ------------------------------------------------------------------
    def lookup_aircraft_model(self, by: str, value: str) -> dict:
        if by == "callsign":
            model = self._aircraft_models.get(value)
            if model:
                return {"candidates": [{"model": model, "confidence": 0.95}]}
            return {"candidates": []}
        if by == "registration":
            for track in self._by_type.get("track", []):
                if track.get("registration") == value:
                    return {"candidates": [{"model": track.get("aircraftModel", "Unknown"), "confidence": 0.90}]}
            return {"candidates": []}
        if by == "kinematics":
            track = self._get_track(value)
            traj = self._get_trajectory(value)
            if not traj:
                return {"candidates": []}
            speeds = [p.get("speed", 0) for p in traj]
            alts = [p.get("alt", 0) for p in traj]
            avg_speed = statistics.mean(speeds)
            avg_alt = statistics.mean(alts)
            candidates = []
            if avg_speed > 450:
                candidates.append({"model": "Su-30MK2", "confidence": 0.7})
                candidates.append({"model": "F-16", "confidence": 0.5})
            elif avg_speed > 350:
                candidates.append({"model": "A321", "confidence": 0.6})
                candidates.append({"model": "B737-800", "confidence": 0.55})
            else:
                candidates.append({"model": "Cessna 172", "confidence": 0.6})
                candidates.append({"model": "ATR72", "confidence": 0.5})
            if avg_alt < 3000:
                candidates.append({"model": "P-3C", "confidence": 0.45})
            return {"candidates": candidates[:3]}
        return {"candidates": []}

    # ------------------------------------------------------------------
    # Tool 7: lookupFlightPlan
    # ------------------------------------------------------------------
    def lookup_flight_plan(self, by: str, value: str) -> dict | None:
        if by == "trackId":
            return self._flight_plans.get(value)
        if by == "callsign":
            for tid, plan in self._flight_plans.items():
                if plan.get("callsign") == value:
                    return plan
        return None

    # ------------------------------------------------------------------
    # Tool 8: findRelated
    # ------------------------------------------------------------------
    def find_related(self, track_id: str, target: str, options: dict | None = None) -> list[dict]:
        self._get_track(track_id)
        traj = self._get_trajectory(track_id)
        opts = options or {}
        radius_km = opts.get("radiusKm", 50)
        top_k = opts.get("topK", 5)

        if not traj:
            return []
        latest = traj[-1]

        if target == "companionAircraft":
            results = []
            for other in self._by_type.get("track", []):
                if other["id"] == track_id:
                    continue
                other_traj = self._get_trajectory(other["id"])
                if not other_traj:
                    continue
                other_latest = other_traj[-1]
                dist = haversine_m(latest["lon"], latest["lat"], other_latest["lon"], other_latest["lat"])
                if dist <= radius_km * 1000:
                    results.append({
                        "id": other["id"],
                        "name": other.get("name", ""),
                        "distanceKm": round(dist / 1000, 1),
                        "geoFeatures": [_point_to_geofeature(other_latest, other["id"])],
                    })
            return sorted(results, key=lambda x: x["distanceKm"])[:top_k]

        if target == "nearbyShips":
            results = []
            for ship in self._by_type.get("ship", []):
                sc = ship["geometry"]["coordinates"]
                dist = haversine_m(latest["lon"], latest["lat"], sc[0], sc[1])
                if dist <= radius_km * 1000:
                    results.append({
                        "id": ship["id"],
                        "name": ship.get("name", ""),
                        "distanceKm": round(dist / 1000, 1),
                        "geoFeatures": [{"type": "point", "geometry": {"lat": sc[1], "lon": sc[0]}, "properties": {"label": ship["name"]}}],
                    })
            return sorted(results, key=lambda x: x["distanceKm"])[:top_k]

        if target == "similarHistoricalTracks":
            results = []
            for ht in self._historical_tracks:
                similarity = _track_similarity(traj, ht.get("trajectory", []))
                results.append({
                    "id": ht["id"],
                    "name": f"{ht['id']} ({ht.get('model', '')})",
                    "similarity": round(similarity, 2),
                    "geoFeatures": [{"type": "line", "geometry": {"coordinates": ht.get("trajectory", [])}, "properties": {"label": ht["id"]}}],
                })
            return sorted(results, key=lambda x: -x["similarity"])[:top_k]

        return []

    # ------------------------------------------------------------------
    # Tool 9: findNearestPOI
    # ------------------------------------------------------------------
    def find_nearest_poi(self, reference: dict, poi_type: str, k: int = 1) -> list[dict]:
        ref_point = self._resolve_reference_point(reference)
        if ref_point is None:
            return []

        pois = self._by_type.get(poi_type, [])
        scored = []
        for poi in pois:
            pc = poi["geometry"]["coordinates"]
            dist = haversine_m(ref_point[0], ref_point[1], pc[0], pc[1])
            scored.append({
                "id": poi["id"],
                "name": poi.get("name", ""),
                "kind": poi.get("kind", poi_type),
                "point": {"lat": pc[1], "lon": pc[0]},
                "distanceKm": round(dist / 1000, 2),
            })
        return sorted(scored, key=lambda x: x["distanceKm"])[:k]

    # ------------------------------------------------------------------
    # Tool 10: computeDistance
    # ------------------------------------------------------------------
    def compute_distance(self, from_ref: dict, to_ref: dict) -> dict:
        from_pt = self._resolve_reference_point(from_ref)
        to_pt = self._resolve_reference_point(to_ref)
        if from_pt is None or to_pt is None:
            return {"distanceKm": -1, "bearing": 0}

        dist = haversine_m(from_pt[0], from_pt[1], to_pt[0], to_pt[1])
        bearing = _bearing(from_pt[1], from_pt[0], to_pt[1], to_pt[0])
        return {"distanceKm": round(dist / 1000, 2), "bearing": round(bearing, 1)}

    # ------------------------------------------------------------------
    # Tool 11: getReferenceData
    # ------------------------------------------------------------------
    def get_reference_data(self, kind: str, filter_opts: dict | None = None) -> list[dict]:
        kind_to_type = {
            "airports": "airport",
            "bases": "base",
            "islands": "island",
            "restrictedZones": "restricted_zone",
            "commonRoutes": "common_route",
        }
        if kind == "aircraftModelProfile":
            name_filter = (filter_opts or {}).get("name", "")
            profiles = {
                "A321": {"maxSpeed": 470, "maxAlt": 39000, "cruiseSpeed": 450, "category": "commercial"},
                "B737-800": {"maxSpeed": 460, "maxAlt": 41000, "cruiseSpeed": 440, "category": "commercial"},
                "Su-30MK2": {"maxSpeed": 1350, "maxAlt": 56000, "cruiseSpeed": 600, "category": "military"},
                "Cessna 172": {"maxSpeed": 130, "maxAlt": 14000, "cruiseSpeed": 110, "category": "light"},
                "P-3C": {"maxSpeed": 400, "maxAlt": 28000, "cruiseSpeed": 330, "category": "military"},
            }
            if name_filter and name_filter in profiles:
                return [{"model": name_filter, **profiles[name_filter]}]
            return [{"model": k, **v} for k, v in profiles.items()]

        obj_type = kind_to_type.get(kind)
        if not obj_type:
            return []
        return [
            {"id": o["id"], "name": o.get("name", ""), "geometry": o.get("geometry")}
            for o in self._by_type.get(obj_type, [])
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_reference_point(self, ref: dict) -> tuple[float, float] | None:
        """Resolve a reference to (lon, lat)."""
        if "lat" in ref and "lon" in ref:
            return (ref["lon"], ref["lat"])
        if "poiName" in ref:
            for objs in self._by_type.values():
                for o in objs:
                    if o.get("name", "").lower() == ref["poiName"].lower():
                        c = o["geometry"]["coordinates"]
                        return (c[0], c[1])
            return None
        if "trackId" in ref:
            traj = self._get_trajectory(ref["trackId"])
            if not traj:
                return None
            when = ref.get("when", "latest")
            if when == "start":
                p = traj[0]
            elif when == "end" or when == "latest":
                p = traj[-1]
            else:
                p = traj[-1]
            return (p["lon"], p["lat"])
        return None


def _point_to_geo(p: dict) -> dict:
    return {"lat": p["lat"], "lon": p["lon"], "alt": p.get("alt"), "timestamp": p.get("timestamp")}


def _point_to_geofeature(p: dict, label: str) -> dict:
    return {
        "type": "point",
        "geometry": {"lat": p["lat"], "lon": p["lon"]},
        "properties": {"label": label, "timestamp": p.get("timestamp")},
    }


def _track_similarity(traj_a: list[dict], coords_b: list[list[float]]) -> float:
    """Simple similarity based on average distance between sampled points."""
    if not traj_a or not coords_b:
        return 0.0
    sample_a = traj_a[::max(len(traj_a) // 10, 1)]
    sample_b = coords_b[::max(len(coords_b) // 10, 1)]
    n = min(len(sample_a), len(sample_b))
    if n == 0:
        return 0.0
    total_dist = sum(
        haversine_m(sample_a[i]["lon"], sample_a[i]["lat"], sample_b[i][0], sample_b[i][1])
        for i in range(n)
    )
    avg_dist_km = total_dist / n / 1000
    return max(0, 1.0 - avg_dist_km / 500)


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2 in degrees."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(rlat2)
    y = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360
