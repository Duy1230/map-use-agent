"""Tests for aircraft mock environment: database, state, and tool handlers."""

from __future__ import annotations

from pathlib import Path

from src.domain.profile import DomainProfile
from src.mock_environment.aircraft_database import AircraftScenarioDatabase, TrackNotFoundError
from src.mock_environment.aircraft_state import AircraftAgentState
from src.mock_environment.aircraft_tools import AIRCRAFT_TOOL_REGISTRY
from src.world_generator.aircraft_generator import generate_aircraft_world


def _make_scenario(seed: int = 42) -> dict:
    return generate_aircraft_world("test_tools", seed=seed, num_tracks=3, num_airports=2, num_bases=1)


def _make_profile() -> DomainProfile:
    return DomainProfile.from_file(Path("profiles/aircraft_track.yaml"))


def test_database_indexes_tracks() -> None:
    scenario = _make_scenario()
    db = AircraftScenarioDatabase(scenario, profile=_make_profile())

    tracks = [obj for obj in db._by_type.get("track", [])]
    assert len(tracks) == 3


def test_get_track_info_returns_identity() -> None:
    scenario = _make_scenario()
    db = AircraftScenarioDatabase(scenario, profile=_make_profile())
    track_id = scenario["objects"]["tracks"][0]["id"]

    info = db.get_track_info(track_id, ["identity"])

    assert info["trackId"] == track_id
    assert "identity" in info
    assert info["identity"]["callSign"]


def test_get_track_info_all_fields() -> None:
    scenario = _make_scenario()
    db = AircraftScenarioDatabase(scenario, profile=_make_profile())
    track_id = scenario["objects"]["tracks"][0]["id"]

    info = db.get_track_info(track_id, ["all"])

    assert "identity" in info
    assert "timespan" in info
    assert "startPoint" in info
    assert "endPoint" in info


def test_get_track_points_returns_trajectory() -> None:
    scenario = _make_scenario()
    db = AircraftScenarioDatabase(scenario, profile=_make_profile())
    track_id = scenario["objects"]["tracks"][0]["id"]

    points = db.get_track_points(track_id)

    assert len(points) > 0
    assert "lat" in points[0]
    assert "lon" in points[0]


def test_get_kinematic_stats_computes_correctly() -> None:
    scenario = _make_scenario()
    db = AircraftScenarioDatabase(scenario, profile=_make_profile())
    track_id = scenario["objects"]["tracks"][0]["id"]

    stats = db.get_kinematic_stats(track_id, "speed")

    assert stats["mean"] > 0
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert "distribution" in stats
    assert "series" in stats


def test_analyze_spatial_restricted_zones() -> None:
    scenario = _make_scenario()
    db = AircraftScenarioDatabase(scenario, profile=_make_profile())
    track_id = scenario["objects"]["tracks"][0]["id"]

    findings = db.analyze_spatial(track_id, "restrictedZones")

    assert isinstance(findings, list)


def test_detect_behavior_no_match_returns_empty() -> None:
    scenario = _make_scenario(seed=1)
    db = AircraftScenarioDatabase(scenario, profile=_make_profile())
    track_id = scenario["objects"]["tracks"][0]["id"]
    track = [t for t in scenario["objects"]["tracks"] if t["id"] == track_id][0]
    tags = track.get("properties", {}).get("behaviorTags", [])

    if "circling" not in tags:
        findings = db.detect_behavior(track_id, "circling")
        assert findings == []


def test_lookup_aircraft_model_by_callsign() -> None:
    scenario = _make_scenario()
    db = AircraftScenarioDatabase(scenario, profile=_make_profile())
    callsign = scenario["objects"]["tracks"][0]["callSign"]

    result = db.lookup_aircraft_model("callsign", callsign)

    assert len(result["candidates"]) > 0
    assert result["candidates"][0]["confidence"] > 0


def test_find_nearest_poi() -> None:
    scenario = _make_scenario()
    db = AircraftScenarioDatabase(scenario, profile=_make_profile())
    track_id = scenario["objects"]["tracks"][0]["id"]

    results = db.find_nearest_poi(
        {"trackId": track_id, "when": "latest"}, "airport", k=2,
    )

    assert len(results) <= 2
    for poi in results:
        assert "distanceKm" in poi
        assert poi["distanceKm"] >= 0


def test_compute_distance_between_pois() -> None:
    scenario = _make_scenario()
    db = AircraftScenarioDatabase(scenario, profile=_make_profile())

    airport = scenario["objects"]["airports"][0]
    coords = airport["geometry"]["coordinates"]

    result = db.compute_distance(
        {"lat": coords[1], "lon": coords[0]},
        {"lat": coords[1] + 1, "lon": coords[0]},
    )

    assert result["distanceKm"] > 0
    assert 0 <= result["bearing"] < 360


def test_get_reference_data_airports() -> None:
    scenario = _make_scenario()
    db = AircraftScenarioDatabase(scenario, profile=_make_profile())

    data = db.get_reference_data("airports")

    assert len(data) == 2


def test_state_map_draw_creates_layer() -> None:
    state = AircraftAgentState(time="2026-05-18T10:00:00+07:00")

    layer_id = state.map_draw(
        [{"type": "point", "geometry": {"lat": 10.0, "lon": 106.0}, "properties": {"label": "test"}}],
    )

    assert layer_id in state.layers
    assert len(state.layers[layer_id]) == 1


def test_state_draw_chart() -> None:
    state = AircraftAgentState(time="2026-05-18T10:00:00+07:00")

    chart_id = state.draw_chart("distribution", {"bins": [1, 2, 3], "counts": [5, 10, 3]}, title="Speed")

    assert len(state.charts) == 1
    assert state.charts[0].chart_id == chart_id
    assert state.charts[0].chart_type == "distribution"


def test_tool_registry_has_14_handlers() -> None:
    assert len(AIRCRAFT_TOOL_REGISTRY) == 14


def test_tool_handler_get_track_info() -> None:
    scenario = _make_scenario()
    db = AircraftScenarioDatabase(scenario, profile=_make_profile())
    state = AircraftAgentState(time=scenario.get("time", ""))
    track_id = scenario["objects"]["tracks"][0]["id"]

    handler = AIRCRAFT_TOOL_REGISTRY["getTrackInfo"]
    result = handler(state, db, {"trackId": track_id, "fields": ["identity"]})

    assert result["trackId"] == track_id
    assert "identity" in result


def test_tool_handler_map_draw() -> None:
    scenario = _make_scenario()
    db = AircraftScenarioDatabase(scenario, profile=_make_profile())
    state = AircraftAgentState(time=scenario.get("time", ""))

    handler = AIRCRAFT_TOOL_REGISTRY["mapDraw"]
    result = handler(state, db, {
        "features": [{"type": "point", "geometry": {"lat": 10, "lon": 106}, "properties": {}}],
        "options": {"autoFocus": True},
    })

    assert "layerId" in result
    assert result["layerId"] in state.layers


def test_track_not_found_raises() -> None:
    scenario = _make_scenario()
    db = AircraftScenarioDatabase(scenario, profile=_make_profile())

    try:
        db.get_track_info("T-NONEXISTENT", ["identity"])
        assert False, "Expected TrackNotFoundError"
    except TrackNotFoundError:
        pass
