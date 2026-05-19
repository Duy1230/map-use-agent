"""Tool handler implementations for the 14 aircraft track analysis tools.

Each handler follows the same signature as the traffic map tools:
    handler(state, db, args) -> object
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.mock_environment.aircraft_database import AircraftScenarioDatabase
from src.mock_environment.aircraft_state import AircraftAgentState


class AircraftToolExecutionError(Exception):
    """Raised when an aircraft tool call fails."""


def _require(args: dict, key: str) -> Any:
    if key not in args:
        raise AircraftToolExecutionError(f"Missing required argument: {key!r}")
    return args[key]


# ------------------------------------------------------------------
# Tier A: Data Access
# ------------------------------------------------------------------

def get_track_info(state: AircraftAgentState, db: AircraftScenarioDatabase, args: dict) -> dict:
    track_id = _require(args, "trackId")
    fields = _require(args, "fields")
    return db.get_track_info(track_id, fields)


def get_track_points(state: AircraftAgentState, db: AircraftScenarioDatabase, args: dict) -> list[dict]:
    track_id = _require(args, "trackId")
    time_range = args.get("timeRange")
    sampling = args.get("sampling", "downsample-10s")
    return db.get_track_points(track_id, time_range, sampling)


# ------------------------------------------------------------------
# Tier B: Analysis
# ------------------------------------------------------------------

def get_kinematic_stats(state: AircraftAgentState, db: AircraftScenarioDatabase, args: dict) -> dict:
    track_id = _require(args, "trackId")
    dimension = _require(args, "dimension")
    return db.get_kinematic_stats(track_id, dimension)


def analyze_spatial(state: AircraftAgentState, db: AircraftScenarioDatabase, args: dict) -> list[dict]:
    track_id = _require(args, "trackId")
    aspect = _require(args, "aspect")
    return db.analyze_spatial(track_id, aspect)


def detect_behavior(state: AircraftAgentState, db: AircraftScenarioDatabase, args: dict) -> list[dict]:
    track_id = _require(args, "trackId")
    pattern = _require(args, "pattern")
    options = args.get("options", {})
    return db.detect_behavior(track_id, pattern, options)


# ------------------------------------------------------------------
# Tier C: Lookup & Cross-Reference
# ------------------------------------------------------------------

def lookup_aircraft_model(state: AircraftAgentState, db: AircraftScenarioDatabase, args: dict) -> dict:
    by = _require(args, "by")
    value = _require(args, "value")
    return db.lookup_aircraft_model(by, value)


def lookup_flight_plan(state: AircraftAgentState, db: AircraftScenarioDatabase, args: dict) -> dict:
    by = _require(args, "by")
    value = _require(args, "value")
    result = db.lookup_flight_plan(by, value)
    return result if result else {"route": None, "origin": None, "destination": None}


def find_related(state: AircraftAgentState, db: AircraftScenarioDatabase, args: dict) -> list[dict]:
    track_id = _require(args, "trackId")
    target = _require(args, "target")
    options = args.get("options", {})
    return db.find_related(track_id, target, options)


# ------------------------------------------------------------------
# Tier D: Geo Utility
# ------------------------------------------------------------------

def find_nearest_poi(state: AircraftAgentState, db: AircraftScenarioDatabase, args: dict) -> list[dict]:
    reference = _require(args, "reference")
    poi_type = _require(args, "poiType")
    k = args.get("k", 1)
    return db.find_nearest_poi(reference, poi_type, k)


def compute_distance(state: AircraftAgentState, db: AircraftScenarioDatabase, args: dict) -> dict:
    from_ref = _require(args, "from")
    to_ref = _require(args, "to")
    return db.compute_distance(from_ref, to_ref)


# ------------------------------------------------------------------
# Tier E: Reference Data
# ------------------------------------------------------------------

def get_reference_data(state: AircraftAgentState, db: AircraftScenarioDatabase, args: dict) -> list[dict]:
    kind = _require(args, "kind")
    filter_opts = args.get("filter", {})
    return db.get_reference_data(kind, filter_opts)


# ------------------------------------------------------------------
# Tier F: Visualization
# ------------------------------------------------------------------

def map_draw(state: AircraftAgentState, db: AircraftScenarioDatabase, args: dict) -> dict:
    features = _require(args, "features")
    options = args.get("options", {})
    layer_id = state.map_draw(features, options)
    return {"layerId": layer_id}


def map_control(state: AircraftAgentState, db: AircraftScenarioDatabase, args: dict) -> dict:
    action = _require(args, "action")
    action_args = args.get("args", {})
    return state.map_control(action, action_args)


def draw_chart(state: AircraftAgentState, db: AircraftScenarioDatabase, args: dict) -> dict:
    chart_type = _require(args, "type")
    data = _require(args, "data")
    title = args.get("title", "")
    chart_id = state.draw_chart(chart_type, data, title)
    return {"chartId": chart_id}


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

AIRCRAFT_TOOL_REGISTRY: dict[str, Callable[[AircraftAgentState, AircraftScenarioDatabase, dict], Any]] = {
    "getTrackInfo": get_track_info,
    "getTrackPoints": get_track_points,
    "getKinematicStats": get_kinematic_stats,
    "analyzeSpatial": analyze_spatial,
    "detectBehavior": detect_behavior,
    "lookupAircraftModel": lookup_aircraft_model,
    "lookupFlightPlan": lookup_flight_plan,
    "findRelated": find_related,
    "findNearestPOI": find_nearest_poi,
    "computeDistance": compute_distance,
    "getReferenceData": get_reference_data,
    "mapDraw": map_draw,
    "mapControl": map_control,
    "drawChart": draw_chart,
}
