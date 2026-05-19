"""Tests for aircraft world generation and trajectory utilities."""

from __future__ import annotations

from datetime import timedelta, timezone

from src.world_generator.aircraft_generator import generate_aircraft_world
from src.world_generator.aircraft_trajectory_utils import (
    generate_aircraft_trajectory,
    inject_circling,
    inject_hovering,
    inject_signal_gap,
    inject_zigzag,
)

TZ_VN = timezone(timedelta(hours=7))


def test_generate_trajectory_returns_correct_fields() -> None:
    traj = generate_aircraft_trajectory(
        108.0, 15.0,
        duration_minutes=10,
        interval_seconds=10,
    )
    assert len(traj) >= 2
    pt = traj[0]
    assert "timestamp" in pt
    assert "lat" in pt
    assert "lon" in pt
    assert "alt" in pt
    assert "speed" in pt
    assert "heading" in pt
    assert "climbRate" in pt


def test_generate_trajectory_has_reasonable_values() -> None:
    import random

    traj = generate_aircraft_trajectory(
        108.0, 15.0,
        start_alt_m=8000.0,
        speed_knots_range=(400.0, 450.0),
        duration_minutes=20,
        interval_seconds=10,
        rng=random.Random(42),
    )
    for pt in traj:
        assert 0 <= pt["heading"] < 360
        assert pt["speed"] > 0
        assert pt["alt"] > 0


def test_inject_circling_modifies_segment() -> None:
    import random

    traj = generate_aircraft_trajectory(
        108.0, 15.0, duration_minutes=30, interval_seconds=10, rng=random.Random(42),
    )
    original_lon = traj[100]["lon"]
    traj = inject_circling(traj, 100, num_circles=1, rng=random.Random(42))
    assert traj[100]["lon"] != original_lon or traj[100]["lat"] != traj[99]["lat"]


def test_inject_hovering_reduces_speed() -> None:
    import random

    traj = generate_aircraft_trajectory(
        108.0, 15.0, duration_minutes=30, interval_seconds=10, rng=random.Random(42),
    )
    traj = inject_hovering(traj, 50, duration_points=20, rng=random.Random(42))
    hover_speeds = [traj[i]["speed"] for i in range(50, 70)]
    assert all(s < 20 for s in hover_speeds)


def test_inject_signal_gap_removes_points() -> None:
    import random

    traj = generate_aircraft_trajectory(
        108.0, 15.0, duration_minutes=60, interval_seconds=10, rng=random.Random(42),
    )
    original_len = len(traj)
    traj, gap_info = inject_signal_gap(traj, 50, gap_points=30)
    assert len(traj) == original_len - 30
    assert gap_info["durationSec"] == 300


def test_inject_zigzag_changes_heading() -> None:
    import random

    traj = generate_aircraft_trajectory(
        108.0, 15.0, duration_minutes=20, interval_seconds=10, rng=random.Random(42),
    )
    base_heading = traj[50]["heading"]
    traj = inject_zigzag(traj, 50, duration_points=10, amplitude_deg=40.0, rng=random.Random(42))
    headings = [traj[i]["heading"] for i in range(50, 60)]
    assert any(abs(h - base_heading) > 5 for h in headings)


def test_generate_aircraft_world_produces_valid_scenario() -> None:
    world = generate_aircraft_world("test_001", seed=42, num_tracks=3, num_airports=2)

    assert world["scenario_id"] == "test_001"
    assert world["coordinate_system"] == "EPSG:4326"
    assert len(world["objects"]["tracks"]) == 3
    assert len(world["objects"]["airports"]) == 2

    track = world["objects"]["tracks"][0]
    assert "id" in track
    assert track["type"] == "track"
    assert "callSign" in track
    assert track["geometry"]["type"] == "LineString"

    assert track["id"] in world["time_series"]
    traj = world["time_series"][track["id"]]["trajectory"]
    assert len(traj) > 0


def test_generate_aircraft_world_includes_all_object_types() -> None:
    world = generate_aircraft_world(
        "test_002", seed=123,
        num_tracks=5, num_airports=3, num_bases=2,
        num_islands=2, num_ships=2, num_restricted_zones=2, num_routes=3,
    )

    assert len(world["objects"]["tracks"]) == 5
    assert len(world["objects"]["airports"]) == 3
    assert len(world["objects"]["bases"]) == 2
    assert len(world["objects"]["islands"]) == 2
    assert len(world["objects"]["ships"]) == 2
    assert len(world["objects"]["restricted_zones"]) == 2
    assert len(world["objects"]["common_routes"]) == 3


def test_aircraft_world_tracks_are_deterministic() -> None:
    w1 = generate_aircraft_world("test_det", seed=99)
    w2 = generate_aircraft_world("test_det", seed=99)

    t1 = w1["objects"]["tracks"][0]
    t2 = w2["objects"]["tracks"][0]
    assert t1["id"] == t2["id"]
    assert t1["callSign"] == t2["callSign"]
