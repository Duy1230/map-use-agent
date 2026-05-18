"""Deterministic geometry helpers for synthetic world generation."""

from __future__ import annotations

import math
import random

from shapely.geometry import LineString, Point, Polygon


# Ho Chi Minh City approximate bounding box (default)
DEFAULT_BBOX = {
    "min_lon": 106.62,
    "max_lon": 106.75,
    "min_lat": 10.75,
    "max_lat": 10.85,
}

EARTH_RADIUS_M = 6_371_000


def meters_to_deg_lat(meters: float) -> float:
    return meters / (EARTH_RADIUS_M * math.pi / 180)


def meters_to_deg_lon(meters: float, lat: float) -> float:
    return meters / (EARTH_RADIUS_M * math.cos(math.radians(lat)) * math.pi / 180)


def random_point(bbox: dict | None = None, rng: random.Random | None = None) -> tuple[float, float]:
    """Return a random (lon, lat) within the bounding box."""
    bbox = bbox or DEFAULT_BBOX
    r = rng or random
    lon = r.uniform(bbox["min_lon"], bbox["max_lon"])
    lat = r.uniform(bbox["min_lat"], bbox["max_lat"])
    return (round(lon, 6), round(lat, 6))


def random_rectangle(
    center: tuple[float, float] | None = None,
    width_m: float = 500.0,
    height_m: float = 400.0,
    bbox: dict | None = None,
    rng: random.Random | None = None,
) -> list[list[float]]:
    """Return GeoJSON Polygon ring (5 points, closed) for a rectangle."""
    if center is None:
        center = random_point(bbox, rng)
    lon, lat = center
    half_w = meters_to_deg_lon(width_m / 2, lat)
    half_h = meters_to_deg_lat(height_m / 2)
    ring = [
        [round(lon - half_w, 6), round(lat - half_h, 6)],
        [round(lon + half_w, 6), round(lat - half_h, 6)],
        [round(lon + half_w, 6), round(lat + half_h, 6)],
        [round(lon - half_w, 6), round(lat + half_h, 6)],
        [round(lon - half_w, 6), round(lat - half_h, 6)],
    ]
    return ring


def random_road_segment(
    bbox: dict | None = None,
    length_m: float = 800.0,
    num_points: int = 4,
    rng: random.Random | None = None,
) -> list[list[float]]:
    """Return a polyline representing a road segment."""
    r = rng or random
    start = random_point(bbox, r)
    heading = r.uniform(0, 360)
    heading_rad = math.radians(heading)

    seg_len = length_m / max(num_points - 1, 1)
    points = [list(start)]
    lon, lat = start
    for _ in range(num_points - 1):
        jitter = r.gauss(0, 0.0001)
        d_lat = meters_to_deg_lat(seg_len) * math.cos(heading_rad) + jitter
        d_lon = meters_to_deg_lon(seg_len, lat) * math.sin(heading_rad) + jitter
        lon = round(lon + d_lon, 6)
        lat = round(lat + d_lat, 6)
        points.append([lon, lat])
    return points


def point_in_polygon(lon: float, lat: float, polygon_coords: list[list[list[float]]]) -> bool:
    pt = Point(lon, lat)
    poly = Polygon(polygon_coords[0])
    return poly.contains(pt)


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine distance in meters."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def line_crosses_trajectory(
    line_coords: list[list[float]], trajectory_coords: list[list[float]]
) -> bool:
    """Check if a trajectory (sequence of points) crosses a line segment."""
    line = LineString(line_coords)
    traj = LineString(trajectory_coords)
    return line.crosses(traj) or line.intersects(traj)
