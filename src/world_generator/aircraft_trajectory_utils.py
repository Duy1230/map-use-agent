"""Generate realistic aircraft flight trajectories with kinematic parameters.

Produces time-series data with lat, lon, alt, speed (knots), heading, and
climbRate for each timestamp.  Supports injecting behavioral anomalies
(circling, hovering, zigzag, U-turn, signal gaps, altitude spikes, etc.)
into otherwise smooth flight paths.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from src.world_generator.geometry_utils import (
    meters_to_deg_lat,
    meters_to_deg_lon,
)

TZ_VN = timezone(timedelta(hours=7))

KNOTS_TO_MPS = 0.514444
KM_TO_M = 1000.0


def generate_aircraft_trajectory(
    start_lon: float,
    start_lat: float,
    *,
    start_alt_m: float = 8000.0,
    heading_deg: float | None = None,
    speed_knots_range: tuple[float, float] = (350.0, 480.0),
    alt_range_m: tuple[float, float] = (5000.0, 12000.0),
    duration_minutes: int = 120,
    interval_seconds: int = 10,
    base_time: datetime | None = None,
    rng: random.Random | None = None,
) -> list[dict]:
    """Generate a straight-line aircraft trajectory with gentle kinematic drift."""
    r = rng or random.Random()
    if heading_deg is None:
        heading_deg = r.uniform(0, 360)
    if base_time is None:
        base_time = datetime(2026, 5, 18, 9, 0, 0, tzinfo=TZ_VN)

    num_points = max(duration_minutes * 60 // interval_seconds, 2)
    speed_kt = r.uniform(*speed_knots_range)

    heading_rad = math.radians(heading_deg)
    lon, lat = start_lon, start_lat
    alt = start_alt_m
    points: list[dict] = []

    for i in range(num_points):
        ts = (
            base_time
            - timedelta(minutes=duration_minutes)
            + timedelta(seconds=i * interval_seconds)
        )
        cur_heading = (heading_deg + r.gauss(0, 0.3)) % 360
        cur_speed = max(speed_kt + r.gauss(0, 2), 50)
        climb_rate = r.gauss(0, 0.5)
        alt = max(300, min(alt + climb_rate * interval_seconds, alt_range_m[1]))

        points.append({
            "timestamp": ts.isoformat(),
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "alt": round(alt, 1),
            "speed": round(cur_speed, 1),
            "heading": round(cur_heading % 360, 1),
            "climbRate": round(climb_rate, 2),
        })

        heading_rad = math.radians(cur_heading)
        real_step = cur_speed * KNOTS_TO_MPS * interval_seconds
        d_lat = meters_to_deg_lat(real_step) * math.cos(heading_rad)
        d_lon = meters_to_deg_lon(real_step, lat) * math.sin(heading_rad)
        lon = round(lon + d_lon, 6)
        lat = round(lat + d_lat, 6)

    return points


def inject_circling(
    points: list[dict],
    start_index: int,
    num_circles: int = 2,
    radius_m: float = 5000.0,
    rng: random.Random | None = None,
) -> list[dict]:
    """Replace a segment with circling behaviour around a centre point."""
    r = rng or random.Random()
    if start_index >= len(points):
        return points

    centre_lat = points[start_index]["lat"]
    centre_lon = points[start_index]["lon"]
    alt = points[start_index]["alt"]
    speed = points[start_index]["speed"]

    circum_m = 2 * math.pi * radius_m
    speed_mps = speed * KNOTS_TO_MPS
    interval = 10
    points_per_circle = max(int(circum_m / (speed_mps * interval)), 12)
    total_circle_pts = points_per_circle * num_circles

    end_index = min(start_index + total_circle_pts, len(points))
    for j, idx in enumerate(range(start_index, end_index)):
        angle = 2 * math.pi * j / points_per_circle
        lat_off = meters_to_deg_lat(radius_m) * math.cos(angle)
        lon_off = meters_to_deg_lon(radius_m, centre_lat) * math.sin(angle)
        heading = (math.degrees(angle) + 90) % 360

        points[idx]["lat"] = round(centre_lat + lat_off, 6)
        points[idx]["lon"] = round(centre_lon + lon_off, 6)
        points[idx]["alt"] = round(alt + r.gauss(0, 20), 1)
        points[idx]["heading"] = round(heading, 1)
        points[idx]["climbRate"] = round(r.gauss(0, 0.3), 2)

    return points


def inject_hovering(
    points: list[dict],
    start_index: int,
    duration_points: int = 60,
    rng: random.Random | None = None,
) -> list[dict]:
    """Replace a segment with near-stationary hovering/loitering."""
    r = rng or random.Random()
    end_index = min(start_index + duration_points, len(points))
    base_lat = points[start_index]["lat"]
    base_lon = points[start_index]["lon"]
    base_alt = points[start_index]["alt"]

    for idx in range(start_index, end_index):
        points[idx]["lat"] = round(base_lat + r.gauss(0, 0.0001), 6)
        points[idx]["lon"] = round(base_lon + r.gauss(0, 0.0001), 6)
        points[idx]["alt"] = round(base_alt + r.gauss(0, 10), 1)
        points[idx]["speed"] = round(max(r.gauss(5, 3), 0), 1)
        points[idx]["heading"] = round(r.uniform(0, 360), 1)
        points[idx]["climbRate"] = round(r.gauss(0, 0.1), 2)

    return points


def inject_zigzag(
    points: list[dict],
    start_index: int,
    duration_points: int = 40,
    amplitude_deg: float = 30.0,
    rng: random.Random | None = None,
) -> list[dict]:
    """Inject rapid heading oscillations (zigzag pattern)."""
    r = rng or random.Random()
    end_index = min(start_index + duration_points, len(points))
    base_heading = points[start_index]["heading"]

    for j, idx in enumerate(range(start_index, end_index)):
        swing = amplitude_deg * math.sin(2 * math.pi * j / 8) + r.gauss(0, 2)
        points[idx]["heading"] = round((base_heading + swing) % 360, 1)

    return points


def inject_uturn(
    points: list[dict],
    index: int,
    rng: random.Random | None = None,
) -> list[dict]:
    """Inject a 180-degree heading reversal at a single point."""
    if index >= len(points):
        return points
    old_heading = points[index]["heading"]
    new_heading = (old_heading + 180) % 360
    points[index]["heading"] = round(new_heading, 1)

    for idx in range(index + 1, min(index + 5, len(points))):
        points[idx]["heading"] = round(new_heading + (random.Random() if rng is None else rng).gauss(0, 2), 1)

    return points


def inject_speed_anomaly(
    points: list[dict],
    start_index: int,
    duration_points: int = 10,
    factor: float = 1.8,
) -> list[dict]:
    """Multiply speed by a factor over a segment (abnormal acceleration)."""
    end_index = min(start_index + duration_points, len(points))
    for idx in range(start_index, end_index):
        points[idx]["speed"] = round(points[idx]["speed"] * factor, 1)

    return points


def inject_altitude_anomaly(
    points: list[dict],
    start_index: int,
    duration_points: int = 15,
    delta_m: float = 3000.0,
) -> list[dict]:
    """Rapid altitude change (climb or dive) over a segment."""
    end_index = min(start_index + duration_points, len(points))
    step = delta_m / max(duration_points, 1)
    for j, idx in enumerate(range(start_index, end_index)):
        points[idx]["alt"] = round(points[idx]["alt"] + step * (j + 1), 1)
        points[idx]["climbRate"] = round(step / 10.0, 2)

    return points


def inject_sustained_low_altitude(
    points: list[dict],
    start_index: int,
    duration_points: int = 120,
    alt_m: float = 150.0,
    rng: random.Random | None = None,
) -> list[dict]:
    """Force the track to fly at sustained low altitude."""
    r = rng or random.Random()
    end_index = min(start_index + duration_points, len(points))
    for idx in range(start_index, end_index):
        points[idx]["alt"] = round(alt_m + r.gauss(0, 20), 1)
        points[idx]["climbRate"] = round(r.gauss(0, 0.05), 2)

    return points


def inject_signal_gap(
    points: list[dict],
    start_index: int,
    gap_points: int = 200,
) -> tuple[list[dict], dict]:
    """Remove points to simulate radar signal loss.

    Returns (modified_points, gap_info_dict).
    """
    end_index = min(start_index + gap_points, len(points))
    if start_index >= len(points) or end_index <= start_index:
        return points, {}

    gap_start_ts = points[start_index]["timestamp"]
    gap_end_ts = points[min(end_index, len(points) - 1)]["timestamp"]
    gap_start_pt = {
        "lat": points[start_index]["lat"],
        "lon": points[start_index]["lon"],
    }
    gap_end_pt = {
        "lat": points[min(end_index, len(points) - 1)]["lat"],
        "lon": points[min(end_index, len(points) - 1)]["lon"],
    }

    modified = points[:start_index] + points[end_index:]

    gap_info = {
        "start": gap_start_ts,
        "end": gap_end_ts,
        "startPoint": gap_start_pt,
        "endPoint": gap_end_pt,
        "durationSec": gap_points * 10,
    }
    return modified, gap_info
