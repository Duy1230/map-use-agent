"""Gate 2 — Static consistency checks (no execution needed)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

VALID_TIME_RANGES = {
    "last_5_minutes", "last_10_minutes", "last_30_minutes", "last_1_hour", "today",
}

VALID_TOOL_NAMES = {
    "get_current_map_state", "get_selected_entity", "resolve_entity",
    "get_object_info", "get_vehicle_trajectory", "get_nearby_objects",
    "get_polygon_area", "get_line_length", "get_objects_inside_polygon",
    "get_objects_crossing_line", "highlight_objects", "draw_polyline",
    "draw_polygon", "draw_marker", "show_popup", "clear_artifacts",
}


def check_static(sample: dict, scenario: dict) -> tuple[bool, list[str]]:
    """Run static consistency checks on a candidate sample.

    Returns (passed, list_of_error_messages).
    """
    errors: list[str] = []

    all_object_ids = {
        obj["id"]
        for cat in scenario.get("objects", {}).values()
        for obj in cat
    }
    object_types = {
        obj["id"]: obj["type"]
        for cat in scenario.get("objects", {}).values()
        for obj in cat
    }

    # Check selected entities exist
    selected = sample.get("initial_state", {}).get("selected", {})
    for slot in ("object", "line", "polygon"):
        entity = selected.get(slot)
        if entity is None:
            continue
        if isinstance(entity, list):
            for e in entity:
                _check_entity_exists(e, all_object_ids, errors)
        elif isinstance(entity, dict):
            _check_entity_exists(entity, all_object_ids, errors)

    # Check gold trace tool names and referenced object IDs
    is_negative = sample.get("expected", {}).get("should_ask_clarification", False)
    for i, step in enumerate(sample.get("gold_trace", [])):
        tool = step.get("tool", "")
        if tool not in VALID_TOOL_NAMES:
            errors.append(f"gold_trace[{i}]: unknown tool {tool!r}")

        args = step.get("args", {})
        for key in ("object_id", "vehicle_id", "polygon_id", "line_id", "target_id"):
            val = args.get(key)
            if val and isinstance(val, str) and not val.startswith("$"):
                if val not in all_object_ids and not is_negative:
                    errors.append(
                        f"gold_trace[{i}].args.{key}={val!r} not in scenario objects"
                    )

        # Type-specific checks
        if tool == "get_vehicle_trajectory":
            vid = args.get("vehicle_id", "")
            if vid in object_types and object_types[vid] != "vehicle":
                errors.append(
                    f"gold_trace[{i}]: get_vehicle_trajectory called on "
                    f"{object_types[vid]!r} ({vid}), expected 'vehicle'"
                )

        if tool == "get_polygon_area" or tool == "get_objects_inside_polygon":
            pid = args.get("polygon_id", "")
            if pid in object_types and object_types[pid] != "polygon":
                errors.append(
                    f"gold_trace[{i}]: {tool} called on "
                    f"{object_types[pid]!r} ({pid}), expected 'polygon'"
                )

        if tool in ("get_line_length", "get_objects_crossing_line"):
            lid = args.get("line_id", "")
            if lid in object_types and object_types[lid] != "line":
                errors.append(
                    f"gold_trace[{i}]: {tool} called on "
                    f"{object_types[lid]!r} ({lid}), expected 'line'"
                )

        # Time range validation
        tr = args.get("time_range")
        if tr and tr not in VALID_TIME_RANGES:
            errors.append(f"gold_trace[{i}]: invalid time_range {tr!r}")

    return (len(errors) == 0, errors)


def _check_entity_exists(entity: dict, all_ids: set[str], errors: list[str]) -> None:
    eid = entity.get("id")
    if eid and eid not in all_ids:
        errors.append(f"Selected entity {eid!r} not found in scenario")
