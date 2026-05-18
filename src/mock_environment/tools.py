"""Canonical tool implementations that operate on MapState + ScenarioDatabase."""

from __future__ import annotations

from typing import Any

from src.mock_environment.database import ScenarioDatabase
from src.mock_environment.state import MapState


class ToolExecutionError(Exception):
    """Raised when a tool call fails (bad args, missing object, etc.)."""


def _require(args: dict, key: str) -> Any:
    if key not in args:
        raise ToolExecutionError(f"Missing required argument: {key!r}")
    return args[key]


# ---------------------------------------------------------------------------
# State tools
# ---------------------------------------------------------------------------

def get_current_map_state(state: MapState, db: ScenarioDatabase, args: dict) -> dict:
    return state.visible_to_agent()


def get_selected_entity(state: MapState, db: ScenarioDatabase, args: dict) -> dict:
    entity_type = args.get("entity_type")
    if entity_type is None:
        return {
            "object": state.selected_object,
            "line": state.selected_line,
            "polygon": state.selected_polygon,
        }
    mapping = {
        "object": state.selected_object,
        "line": state.selected_line,
        "polygon": state.selected_polygon,
    }
    if entity_type not in mapping:
        raise ToolExecutionError(f"Invalid entity_type: {entity_type!r}")
    return {"entity": mapping[entity_type]}


def resolve_entity(state: MapState, db: ScenarioDatabase, args: dict) -> dict:
    reference = _require(args, "reference")
    expected_type = args.get("expected_type")
    result = db.resolve_reference(reference, expected_type)
    if result is None:
        return {"entity": None, "confidence": 0.0}
    return {"entity": {"id": result["id"], "type": result["type"], "name": result["name"]}, "confidence": 1.0}


# ---------------------------------------------------------------------------
# Object tools
# ---------------------------------------------------------------------------

def get_object_info(state: MapState, db: ScenarioDatabase, args: dict) -> dict:
    object_id = _require(args, "object_id")
    obj = db.get_object(object_id)
    return {
        "id": obj["id"],
        "type": obj["type"],
        "name": obj["name"],
        "geometry": obj["geometry"],
        "properties": obj.get("properties", {}),
    }


def get_vehicle_trajectory(state: MapState, db: ScenarioDatabase, args: dict) -> dict:
    vehicle_id = _require(args, "vehicle_id")
    time_range = _require(args, "time_range")
    points = db.get_vehicle_trajectory(vehicle_id, time_range)
    return {
        "vehicle_id": vehicle_id,
        "points": [pt["coordinates"] for pt in points],
        "timestamps": [pt["timestamp"] for pt in points],
        "time_range": time_range,
    }


def get_nearby_objects(state: MapState, db: ScenarioDatabase, args: dict) -> dict:
    object_id = _require(args, "object_id")
    object_type = _require(args, "object_type")
    radius_meters = _require(args, "radius_meters")
    results = db.get_nearby_objects(object_id, object_type, float(radius_meters))
    return {
        "center_object_id": object_id,
        "results": [
            {"id": r["id"], "type": r["type"], "name": r["name"], "distance_m": r.get("_distance_m")}
            for r in results
        ],
        "radius_meters": radius_meters,
    }


# ---------------------------------------------------------------------------
# Polygon / line tools
# ---------------------------------------------------------------------------

def get_polygon_area(state: MapState, db: ScenarioDatabase, args: dict) -> dict:
    polygon_id = _require(args, "polygon_id")
    area = db.get_polygon_area_sqm(polygon_id)
    return {"polygon_id": polygon_id, "area_sqm": round(area, 2)}


def get_line_length(state: MapState, db: ScenarioDatabase, args: dict) -> dict:
    line_id = _require(args, "line_id")
    length = db.get_line_length_m(line_id)
    return {"line_id": line_id, "length_meters": length}


def get_objects_inside_polygon(state: MapState, db: ScenarioDatabase, args: dict) -> dict:
    polygon_id = _require(args, "polygon_id")
    object_type = _require(args, "object_type")
    results = db.get_objects_inside_polygon(polygon_id, object_type)
    return {
        "polygon_id": polygon_id,
        "results": [
            {"id": r["id"], "type": r["type"], "name": r["name"]}
            for r in results
        ],
    }


def get_objects_crossing_line(state: MapState, db: ScenarioDatabase, args: dict) -> dict:
    line_id = _require(args, "line_id")
    object_type = _require(args, "object_type")
    time_range = _require(args, "time_range")
    results = db.get_objects_crossing_line(line_id, object_type, time_range)
    return {
        "line_id": line_id,
        "results": [
            {"id": r["id"], "type": r["type"], "name": r["name"]}
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Map action tools
# ---------------------------------------------------------------------------

def highlight_objects(state: MapState, db: ScenarioDatabase, args: dict) -> dict:
    object_ids = _require(args, "object_ids")
    for oid in object_ids:
        db.get_object(oid)  # validate existence
    state.highlighted_objects.update(object_ids)
    return {"highlighted": sorted(state.highlighted_objects)}


def draw_polyline(state: MapState, db: ScenarioDatabase, args: dict) -> dict:
    points = _require(args, "points")
    label = _require(args, "label")
    source_object_id = args.get("source_object_id")
    aid = state.add_artifact(
        artifact_type="polyline",
        geometry={"type": "LineString", "coordinates": points},
        label=label,
        source_object_id=source_object_id,
    )
    return {"artifact_id": aid}


def draw_polygon(state: MapState, db: ScenarioDatabase, args: dict) -> dict:
    geometry = _require(args, "geometry")
    label = _require(args, "label")
    aid = state.add_artifact(
        artifact_type="polygon",
        geometry=geometry,
        label=label,
    )
    return {"artifact_id": aid}


def draw_marker(state: MapState, db: ScenarioDatabase, args: dict) -> dict:
    coordinates = _require(args, "coordinates")
    label = _require(args, "label")
    aid = state.add_artifact(
        artifact_type="marker",
        geometry={"type": "Point", "coordinates": coordinates},
        label=label,
    )
    return {"artifact_id": aid}


def show_popup(state: MapState, db: ScenarioDatabase, args: dict) -> dict:
    target_id = _require(args, "target_id")
    content = _require(args, "content")
    db.get_object(target_id)  # validate existence
    pid = state.add_popup(target_id, content)
    return {"popup_id": pid}


def clear_artifacts(state: MapState, db: ScenarioDatabase, args: dict) -> dict:
    scope = _require(args, "scope")
    valid = {"all", "polylines", "polygons", "markers", "popups", "highlights"}
    if scope not in valid:
        raise ToolExecutionError(f"Invalid scope: {scope!r}. Must be one of {valid}")
    state.clear(scope)
    return {"cleared": scope}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, callable] = {
    "get_current_map_state": get_current_map_state,
    "get_selected_entity": get_selected_entity,
    "resolve_entity": resolve_entity,
    "get_object_info": get_object_info,
    "get_vehicle_trajectory": get_vehicle_trajectory,
    "get_nearby_objects": get_nearby_objects,
    "get_polygon_area": get_polygon_area,
    "get_line_length": get_line_length,
    "get_objects_inside_polygon": get_objects_inside_polygon,
    "get_objects_crossing_line": get_objects_crossing_line,
    "highlight_objects": highlight_objects,
    "draw_polyline": draw_polyline,
    "draw_polygon": draw_polygon,
    "draw_marker": draw_marker,
    "show_popup": show_popup,
    "clear_artifacts": clear_artifacts,
}
