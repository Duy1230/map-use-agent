"""Gate 2 — Static consistency checks (no execution needed)."""

from __future__ import annotations

import logging

from src.domain.scenario import iter_scenario_objects
from src.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

VALID_TIME_RANGES = {
    "last_5_minutes",
    "last_10_minutes",
    "last_30_minutes",
    "last_1_hour",
    "today",
}

VALID_TOOL_NAMES = {
    "get_current_map_state",
    "get_selected_entity",
    "resolve_entity",
    "get_object_info",
    "get_vehicle_trajectory",
    "get_nearby_objects",
    "get_polygon_area",
    "get_line_length",
    "get_objects_inside_polygon",
    "get_objects_crossing_line",
    "highlight_objects",
    "draw_polyline",
    "draw_polygon",
    "draw_marker",
    "show_popup",
    "clear_artifacts",
}


def check_static(
    sample: dict,
    scenario: dict,
    *,
    context: PipelineContext | None = None,
) -> tuple[bool, list[str]]:
    """Run static consistency checks on a candidate sample.

    Returns (passed, list_of_error_messages).
    """
    errors: list[str] = []

    all_object_ids = {
        obj["id"] for obj in iter_scenario_objects(scenario, context.profile if context else None)
    }
    object_types = {
        obj["id"]: obj["type"]
        for obj in iter_scenario_objects(scenario, context.profile if context else None)
    }
    valid_tool_names = set(context.tool_names) if context else VALID_TOOL_NAMES
    valid_time_ranges = set(context.time_ranges) if context else VALID_TIME_RANGES
    selection_slots = context.selection_slots if context else ["object", "line", "polygon"]
    object_reference_args = (
        context.profile.object_reference_args
        if context and context.profile.object_reference_args
        else ["object_id", "vehicle_id", "polygon_id", "line_id", "target_id"]
    )
    type_constraints = context.profile.tool_type_constraints if context else {}

    # Check selected entities exist
    selected = sample.get("initial_state", {}).get("selected", {})
    for slot in selection_slots:
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
        if tool not in valid_tool_names:
            errors.append(f"gold_trace[{i}]: unknown tool {tool!r}")

        args = step.get("args", {})
        for key in object_reference_args:
            val = args.get(key)
            if val and isinstance(val, str) and not val.startswith("$"):
                if val not in all_object_ids and not is_negative:
                    errors.append(f"gold_trace[{i}].args.{key}={val!r} not in scenario objects")
        if context:
            conditional_refs = context.profile.conditional_object_reference_args.get(tool, {})
            for arg_name, rule in conditional_refs.items():
                if not _conditional_reference_applies(args, rule):
                    continue
                val = args.get(arg_name)
                if not val or not isinstance(val, str) or val.startswith("$"):
                    continue
                if val not in all_object_ids and not is_negative:
                    errors.append(
                        f"gold_trace[{i}].args.{arg_name}={val!r} not in scenario objects"
                    )
                    continue
                expected_type = rule.get("expected_type")
                if expected_type and val in object_types and object_types[val] != expected_type:
                    errors.append(
                        f"gold_trace[{i}]: {tool} called on "
                        f"{object_types[val]!r} ({val}), expected {expected_type!r}"
                    )

        # Type-specific checks derived from the profile.
        constraints = type_constraints.get(tool, {})
        if not constraints and context is None:
            constraints = _legacy_type_constraints(tool)
        for arg_name, expected_type in constraints.items():
            object_id = args.get(arg_name, "")
            if object_id in object_types and object_types[object_id] != expected_type:
                errors.append(
                    f"gold_trace[{i}]: {tool} called on "
                    f"{object_types[object_id]!r} ({object_id}), expected {expected_type!r}"
                )

        # Time range validation
        tr = args.get("time_range")
        enum = context.profile.parameter_enum(tool, "time_range") if context else None
        if tr and tr not in valid_time_ranges and (enum is None or tr not in enum):
            errors.append(f"gold_trace[{i}]: invalid time_range {tr!r}")

    return (len(errors) == 0, errors)


def _check_entity_exists(entity: dict, all_ids: set[str], errors: list[str]) -> None:
    eid = entity.get("id")
    if eid and eid not in all_ids:
        errors.append(f"Selected entity {eid!r} not found in scenario")


def _conditional_reference_applies(args: dict, rule: dict) -> bool:
    condition = rule.get("when", {})
    if not condition:
        return True
    return all(args.get(key) == value for key, value in condition.items())


def _legacy_type_constraints(tool: str) -> dict[str, str]:
    if tool == "get_vehicle_trajectory":
        return {"vehicle_id": "vehicle"}
    if tool in ("get_polygon_area", "get_objects_inside_polygon"):
        return {"polygon_id": "polygon"}
    if tool in ("get_line_length", "get_objects_crossing_line"):
        return {"line_id": "line"}
    return {}
