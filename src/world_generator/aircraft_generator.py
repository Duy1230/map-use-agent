"""Deterministic synthetic aircraft world generator.

Produces scenarios with tracks (including kinematic time-series), POIs
(airports, bases, islands, ships), restricted zones, and common air routes.
Behavioral anomalies are injected deterministically based on seed.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from src.world_generator.geometry_utils import (
    random_point,
    random_rectangle,
)
from src.world_generator.aircraft_trajectory_utils import (
    TZ_VN,
    generate_aircraft_trajectory,
    inject_altitude_anomaly,
    inject_circling,
    inject_hovering,
    inject_signal_gap,
    inject_speed_anomaly,
    inject_sustained_low_altitude,
    inject_uturn,
    inject_zigzag,
)

# Vietnam / South China Sea bounding box for aircraft scenarios
AIRCRAFT_BBOX = {
    "min_lon": 105.5,
    "max_lon": 115.0,
    "min_lat": 8.0,
    "max_lat": 21.0,
}

_TRACK_TYPES = ["HKDD", "transit", "VIP", "hostile", "unknown"]
_CALLSIGN_PREFIXES = ["VN", "QV", "BL", "VJ", "UN", "CZ", "MH", "SQ"]
_AIRCRAFT_MODELS = [
    "A321", "B737-800", "B787-9", "A350-900", "ATR72",
    "C-130", "Su-30MK2", "P-3C", "Cessna 172", "B737-MAX",
]
_AIRPORT_NAMES = [
    "Nội Bài", "Tân Sơn Nhất", "Đà Nẵng", "Cam Ranh", "Phú Quốc",
    "Cát Bi", "Vinh", "Huế", "Pleiku", "Liên Khương",
]
_BASE_NAMES = [
    "Căn cứ Biên Hòa", "Căn cứ Phan Rang", "Căn cứ Đà Nẵng",
    "Căn cứ Cam Ranh", "Căn cứ Sao Vàng", "Căn cứ Yên Bái",
]
_ISLAND_NAMES = [
    "Cồn Cỏ", "Lý Sơn", "Phú Quý", "Hoàng Sa", "Trường Sa",
    "Côn Đảo", "Thổ Chu", "Bạch Long Vĩ",
]
_SHIP_PREFIXES = ["HQ", "CSB", "KN"]

_ROUTE_NAMES = [
    "W1 (HAN-DAD)", "A1 (DAD-SGN)", "R474 (SGN-CXR)",
    "G580 (HAN-HPH)", "B209 (HAN-VDO)",
]

_BEHAVIOR_PATTERNS = [
    "circling", "hovering", "zigzag", "uturn",
    "speed_anomaly", "altitude_anomaly", "sustained_low",
    "signal_gap",
]


def _make_callsign(idx: int, rng: random.Random) -> str:
    prefix = rng.choice(_CALLSIGN_PREFIXES)
    return f"{prefix}{rng.randint(100, 999)}"


def _make_registration(idx: int, rng: random.Random) -> str:
    letters = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))
    return f"VN-{letters}"


def generate_aircraft_world(
    scenario_id: str,
    *,
    num_tracks: int = 5,
    num_airports: int = 4,
    num_bases: int = 3,
    num_islands: int = 3,
    num_ships: int = 2,
    num_restricted_zones: int = 2,
    num_routes: int = 3,
    bbox: dict | None = None,
    trajectory_duration_min: int = 120,
    seed: int | None = None,
    base_time: datetime | None = None,
) -> dict:
    """Generate a single synthetic aircraft scenario."""
    rng = random.Random(seed)
    bbox = bbox or AIRCRAFT_BBOX
    if base_time is None:
        base_time = datetime(2026, 5, 18, 10, 0, 0, tzinfo=TZ_VN)

    tracks = []
    track_time_series: dict[str, dict] = {}
    flight_plans: dict[str, dict] = {}
    aircraft_models: dict[str, str] = {}

    for i in range(1, num_tracks + 1):
        tid = f"T-{rng.randint(1000, 9999)}"
        lon, lat = random_point(bbox, rng)
        track_type = rng.choice(_TRACK_TYPES)
        callsign = _make_callsign(i, rng)
        registration = _make_registration(i, rng)
        model = rng.choice(_AIRCRAFT_MODELS)
        aircraft_models[callsign] = model

        speed_range = (350.0, 480.0) if track_type != "hostile" else (400.0, 600.0)
        alt_start = rng.uniform(5000, 11000)

        traj = generate_aircraft_trajectory(
            lon, lat,
            start_alt_m=alt_start,
            speed_knots_range=speed_range,
            alt_range_m=(3000.0, 12000.0),
            duration_minutes=trajectory_duration_min,
            interval_seconds=10,
            base_time=base_time,
            rng=rng,
        )

        behavior_tags: list[str] = []
        signal_gaps: list[dict] = []

        if rng.random() < 0.4 and len(traj) > 200:
            pattern = rng.choice(_BEHAVIOR_PATTERNS)
            start_idx = rng.randint(50, len(traj) // 2)
            behavior_tags.append(pattern)

            if pattern == "circling":
                traj = inject_circling(traj, start_idx, num_circles=rng.randint(1, 3), rng=rng)
            elif pattern == "hovering":
                traj = inject_hovering(traj, start_idx, duration_points=rng.randint(30, 80), rng=rng)
            elif pattern == "zigzag":
                traj = inject_zigzag(traj, start_idx, duration_points=rng.randint(20, 50), rng=rng)
            elif pattern == "uturn":
                traj = inject_uturn(traj, start_idx, rng=rng)
            elif pattern == "speed_anomaly":
                traj = inject_speed_anomaly(traj, start_idx, duration_points=rng.randint(5, 20), factor=rng.uniform(1.5, 2.5))
            elif pattern == "altitude_anomaly":
                traj = inject_altitude_anomaly(traj, start_idx, duration_points=rng.randint(10, 25), delta_m=rng.uniform(2000, 4000))
            elif pattern == "sustained_low":
                traj = inject_sustained_low_altitude(traj, start_idx, duration_points=rng.randint(60, 150), alt_m=rng.uniform(80, 300), rng=rng)
            elif pattern == "signal_gap":
                gap_pts = rng.randint(180, 360)
                traj, gap_info = inject_signal_gap(traj, start_idx, gap_points=gap_pts)
                if gap_info:
                    signal_gaps.append(gap_info)

        start_pt = traj[0] if traj else {}
        end_pt = traj[-1] if traj else {}

        tracks.append({
            "id": tid,
            "type": "track",
            "name": f"Track {tid}",
            "callSign": callsign,
            "registration": registration,
            "targetType": track_type,
            "aircraftModel": model,
            "geometry": {
                "type": "LineString",
                "coordinates": [[p["lon"], p["lat"]] for p in traj],
            },
            "properties": {
                "startTime": start_pt.get("timestamp", ""),
                "endTime": end_pt.get("timestamp", ""),
                "signalGaps": signal_gaps,
                "behaviorTags": behavior_tags,
            },
        })
        track_time_series[tid] = {"trajectory": traj}

        if rng.random() < 0.6:
            flight_plans[tid] = {
                "callsign": callsign,
                "origin": rng.choice(_AIRPORT_NAMES),
                "destination": rng.choice(_AIRPORT_NAMES),
                "route": [[p["lon"], p["lat"]] for p in traj[::max(len(traj) // 5, 1)]],
            }

    airports = []
    for i in range(num_airports):
        name = _AIRPORT_NAMES[i % len(_AIRPORT_NAMES)]
        lon, lat = random_point(bbox, rng)
        airports.append({
            "id": f"airport_{i + 1:02d}",
            "type": "airport",
            "name": name,
            "kind": "airport",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {},
        })

    bases = []
    for i in range(num_bases):
        name = _BASE_NAMES[i % len(_BASE_NAMES)]
        lon, lat = random_point(bbox, rng)
        bases.append({
            "id": f"base_{i + 1:02d}",
            "type": "base",
            "name": name,
            "kind": "base",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {},
        })

    islands = []
    for i in range(num_islands):
        name = _ISLAND_NAMES[i % len(_ISLAND_NAMES)]
        lon, lat = random_point(bbox, rng)
        islands.append({
            "id": f"island_{i + 1:02d}",
            "type": "island",
            "name": name,
            "kind": "island",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {},
        })

    ships = []
    for i in range(num_ships):
        prefix = rng.choice(_SHIP_PREFIXES)
        lon, lat = random_point(bbox, rng)
        ships.append({
            "id": f"ship_{i + 1:02d}",
            "type": "ship",
            "name": f"{prefix}-{rng.randint(100, 999)}",
            "kind": "ship",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"heading_deg": round(rng.uniform(0, 360), 1)},
        })

    restricted_zones = []
    for i in range(num_restricted_zones):
        centre = random_point(bbox, rng)
        w = rng.uniform(30_000, 80_000)
        h = rng.uniform(30_000, 80_000)
        ring = random_rectangle(centre, width_m=w, height_m=h, rng=rng)
        restricted_zones.append({
            "id": f"rz_{i + 1:02d}",
            "type": "restricted_zone",
            "name": f"Vùng cấm {i + 1:02d}",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {},
        })

    common_routes = []
    for i in range(num_routes):
        name = _ROUTE_NAMES[i % len(_ROUTE_NAMES)]
        p1 = random_point(bbox, rng)
        p2 = random_point(bbox, rng)
        mid_lon = (p1[0] + p2[0]) / 2 + rng.gauss(0, 0.5)
        mid_lat = (p1[1] + p2[1]) / 2 + rng.gauss(0, 0.3)
        common_routes.append({
            "id": f"route_{i + 1:02d}",
            "type": "common_route",
            "name": name,
            "geometry": {
                "type": "LineString",
                "coordinates": [list(p1), [round(mid_lon, 6), round(mid_lat, 6)], list(p2)],
            },
            "properties": {},
        })

    historical_tracks: list[dict] = []
    for i in range(rng.randint(2, 5)):
        lon, lat = random_point(bbox, rng)
        htraj = generate_aircraft_trajectory(
            lon, lat,
            duration_minutes=60,
            interval_seconds=30,
            base_time=base_time - timedelta(days=rng.randint(1, 90)),
            rng=rng,
        )
        historical_tracks.append({
            "id": f"HT-{rng.randint(1000, 9999)}",
            "model": rng.choice(_AIRCRAFT_MODELS),
            "trajectory": [[p["lon"], p["lat"]] for p in htraj],
            "similarity": 0.0,
        })

    return {
        "scenario_id": scenario_id,
        "coordinate_system": "EPSG:4326",
        "time": base_time.isoformat(),
        "objects": {
            "tracks": tracks,
            "airports": airports,
            "bases": bases,
            "islands": islands,
            "ships": ships,
            "restricted_zones": restricted_zones,
            "common_routes": common_routes,
        },
        "time_series": track_time_series,
        "flight_plans": flight_plans,
        "aircraft_models": aircraft_models,
        "historical_tracks": historical_tracks,
    }


def generate_aircraft_worlds(
    count: int = 50,
    *,
    output_dir: str | Path = "scenarios",
    base_seed: int = 42,
    small_fraction: float = 0.3,
    medium_fraction: float = 0.5,
) -> list[Path]:
    """Generate *count* aircraft worlds and write them as JSON.

    Returns the list of written file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for i in range(1, count + 1):
        sid = f"aircraft_synthetic_{i:03d}"
        r = i / count
        if r <= small_fraction:
            cfg = dict(num_tracks=3, num_airports=2, num_bases=2, num_islands=2,
                        num_ships=1, num_restricted_zones=1, num_routes=2)
        elif r <= small_fraction + medium_fraction:
            cfg = dict(num_tracks=5, num_airports=4, num_bases=3, num_islands=3,
                        num_ships=2, num_restricted_zones=2, num_routes=3)
        else:
            cfg = dict(num_tracks=8, num_airports=6, num_bases=4, num_islands=5,
                        num_ships=4, num_restricted_zones=3, num_routes=5)

        world = generate_aircraft_world(sid, seed=base_seed + i, **cfg)
        p = out / f"{sid}.json"
        p.write_text(json.dumps(world, indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(p)

    return paths
