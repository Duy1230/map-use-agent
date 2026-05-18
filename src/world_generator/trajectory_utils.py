"""Generate realistic vehicle trajectories as time-series data."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from src.world_generator.geometry_utils import meters_to_deg_lat, meters_to_deg_lon


TZ_VN = timezone(timedelta(hours=7))


def generate_trajectory(
    start_lon: float,
    start_lat: float,
    *,
    heading_deg: float | None = None,
    speed_kmh_range: tuple[float, float] = (30.0, 60.0),
    duration_minutes: int = 30,
    interval_seconds: int = 300,
    base_time: datetime | None = None,
    rng: random.Random | None = None,
) -> list[dict]:
    """Generate a trajectory as a list of {timestamp, coordinates} dicts.

    Points advance roughly along *heading_deg* (with jitter) at a speed
    drawn uniformly from *speed_kmh_range*, sampled every *interval_seconds*.
    """
    r = rng or random
    if heading_deg is None:
        heading_deg = r.uniform(0, 360)
    if base_time is None:
        base_time = datetime.now(TZ_VN).replace(second=0, microsecond=0)

    num_points = max(duration_minutes * 60 // interval_seconds, 2)
    speed_kmh = r.uniform(*speed_kmh_range)
    speed_mps = speed_kmh * 1000 / 3600
    step_m = speed_mps * interval_seconds

    heading_rad = math.radians(heading_deg)
    lon, lat = start_lon, start_lat
    points = []

    for i in range(num_points):
        ts = base_time - timedelta(minutes=duration_minutes) + timedelta(seconds=i * interval_seconds)
        points.append({
            "timestamp": ts.isoformat(),
            "coordinates": [round(lon, 6), round(lat, 6)],
        })
        jitter_rad = r.gauss(0, 0.05)
        d_lat = meters_to_deg_lat(step_m) * math.cos(heading_rad + jitter_rad)
        d_lon = meters_to_deg_lon(step_m, lat) * math.sin(heading_rad + jitter_rad)
        lon = round(lon + d_lon, 6)
        lat = round(lat + d_lat, 6)

    return points
