"""Gold plan generator — maps blueprints to canonical tool traces.

Uses deterministic mapping for well-known task types and falls back to LLM
for complex multi-step plans.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.llm_backend.base import LLMBackend
from src.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic plan templates keyed by task_type
# ---------------------------------------------------------------------------


def _plan_draw_vehicle_trajectory(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    vehicle_id = target.get("object_id")
    time_range = bp.get("constraints", {}).get("time_range", "last_10_minutes")

    steps: list[dict] = [{"tool": "get_current_map_state", "args": {}}]

    if target.get("reference_mode") == "selected_object":
        steps.append({"tool": "get_selected_entity", "args": {"entity_type": "object"}})

    steps.append(
        {
            "tool": "get_vehicle_trajectory",
            "args": {"vehicle_id": vehicle_id, "time_range": time_range},
        }
    )
    steps.append(
        {
            "tool": "draw_polyline",
            "args": {
                "points": "$get_vehicle_trajectory.points",
                "label": f"Trajectory of {vehicle_id}",
                "source_object_id": vehicle_id,
            },
        }
    )
    return steps


def _plan_show_object_info(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    object_id = target.get("object_id")
    steps: list[dict] = [{"tool": "get_current_map_state", "args": {}}]
    if target.get("reference_mode") == "selected_object":
        steps.append({"tool": "get_selected_entity", "args": {"entity_type": "object"}})
    steps.append({"tool": "get_object_info", "args": {"object_id": object_id}})
    steps.append(
        {
            "tool": "show_popup",
            "args": {"target_id": object_id, "content": f"Info for {object_id}"},
        }
    )
    return steps


def _plan_highlight_selected(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    object_id = target.get("object_id")
    return [
        {"tool": "get_current_map_state", "args": {}},
        {"tool": "get_selected_entity", "args": {"entity_type": "object"}},
        {"tool": "highlight_objects", "args": {"object_ids": [object_id]}},
    ]


def _plan_objects_inside_polygon(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    polygon_id = target.get("object_id")
    obj_type = bp.get("constraints", {}).get("object_type", "vehicle")
    steps: list[dict] = [{"tool": "get_current_map_state", "args": {}}]
    if target.get("reference_mode") == "selected_object":
        steps.append({"tool": "get_selected_entity", "args": {"entity_type": "polygon"}})
    steps.append(
        {
            "tool": "get_objects_inside_polygon",
            "args": {"polygon_id": polygon_id, "object_type": obj_type},
        }
    )
    steps.append(
        {
            "tool": "highlight_objects",
            "args": {"object_ids": "$get_objects_inside_polygon.results"},
        }
    )
    return steps


def _plan_objects_crossing_line(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    line_id = target.get("object_id")
    time_range = bp.get("constraints", {}).get("time_range", "last_10_minutes")
    obj_type = bp.get("constraints", {}).get("object_type", "vehicle")
    return [
        {"tool": "get_current_map_state", "args": {}},
        {
            "tool": "get_objects_crossing_line",
            "args": {
                "line_id": line_id,
                "object_type": obj_type,
                "time_range": time_range,
            },
        },
    ]


def _plan_polygon_area(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    polygon_id = target.get("object_id")
    return [
        {"tool": "get_current_map_state", "args": {}},
        {"tool": "get_polygon_area", "args": {"polygon_id": polygon_id}},
    ]


def _plan_line_length(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    line_id = target.get("object_id")
    return [
        {"tool": "get_current_map_state", "args": {}},
        {"tool": "get_line_length", "args": {"line_id": line_id}},
    ]


def _plan_nearby_objects(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    object_id = target.get("object_id")
    obj_type = bp.get("constraints", {}).get("object_type", "camera")
    radius = bp.get("constraints", {}).get("radius_meters", 1000)
    return [
        {"tool": "get_current_map_state", "args": {}},
        {
            "tool": "get_nearby_objects",
            "args": {
                "object_id": object_id,
                "object_type": obj_type,
                "radius_meters": radius,
            },
        },
        {"tool": "highlight_objects", "args": {"object_ids": "$get_nearby_objects.results"}},
    ]


_DETERMINISTIC_PLANS: dict[str, callable] = {
    "draw_vehicle_trajectory": _plan_draw_vehicle_trajectory,
    "show_object_info": _plan_show_object_info,
    "highlight_selected_entity": _plan_highlight_selected,
    "objects_inside_polygon": _plan_objects_inside_polygon,
    "objects_crossing_line": _plan_objects_crossing_line,
    "polygon_area": _plan_polygon_area,
    "line_length": _plan_line_length,
    "nearby_objects": _plan_nearby_objects,
}


# ---------------------------------------------------------------------------
# Aircraft deterministic plan templates
# ---------------------------------------------------------------------------


def _aircraft_plan_routine_surveillance(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    track_id = target.get("object_id")
    return [
        {"tool": "getTrackInfo", "args": {"trackId": track_id, "fields": ["identity", "latestPoint"]}},
        {
            "tool": "mapDraw",
            "args": {
                "features": [
                    {
                        "type": "point",
                        "geometry": "$getTrackInfo.latestPoint",
                        "properties": {"label": "Latest position"},
                    }
                ],
                "options": {"autoFocus": True},
            },
        },
    ]


def _aircraft_plan_target_identification(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    track_id = target.get("object_id")
    return [
        {"tool": "getTrackInfo", "args": {"trackId": track_id, "fields": ["identity"]}},
    ]


def _aircraft_plan_threat_detection(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    track_id = target.get("object_id")
    return [
        {"tool": "analyzeSpatial", "args": {"trackId": track_id, "aspect": "restrictedZones"}},
        {"tool": "mapDraw", "args": {"features": "$analyzeSpatial.geoFeatures", "options": {"autoFocus": True}}},
    ]


def _aircraft_plan_behavioral_anomaly(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    track_id = target.get("object_id")
    pattern = bp.get("constraints", {}).get("pattern", "hovering")
    return [
        {"tool": "detectBehavior", "args": {"trackId": track_id, "pattern": pattern}},
        {"tool": "mapDraw", "args": {"features": "$detectBehavior.geoFeatures", "options": {"autoFocus": True}}},
    ]


def _aircraft_plan_decision_support(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    track_id = target.get("object_id")
    poi_type = bp.get("constraints", {}).get("poiType", "base")
    return [
        {"tool": "findNearestPOI", "args": {"reference": {"trackId": track_id, "when": "latest"}, "poiType": poi_type, "k": 3}},
        {"tool": "mapDraw", "args": {"features": "$findNearestPOI.geoFeatures", "options": {"autoFocus": True}}},
    ]


def _aircraft_plan_coordination(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    track_id = target.get("object_id")
    return [
        {"tool": "findRelated", "args": {"trackId": track_id, "target": "companionAircraft"}},
        {"tool": "mapDraw", "args": {"features": "$findRelated.geoFeatures", "options": {"autoFocus": True}}},
    ]


def _aircraft_plan_post_event_reporting(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    track_id = target.get("object_id")
    return [
        {"tool": "getTrackInfo", "args": {"trackId": track_id, "fields": ["identity", "timespan", "startPoint", "endPoint", "signalGaps"]}},
        {"tool": "analyzeSpatial", "args": {"trackId": track_id, "aspect": "notablePoints"}},
    ]


def _aircraft_plan_spatial_flexible(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    track_id = target.get("object_id")
    aspect = bp.get("constraints", {}).get("aspect", "overflownAreas")
    return [
        {"tool": "analyzeSpatial", "args": {"trackId": track_id, "aspect": aspect}},
    ]


_AIRCRAFT_DETERMINISTIC_PLANS: dict[str, callable] = {
    "routine_surveillance": _aircraft_plan_routine_surveillance,
    "target_identification": _aircraft_plan_target_identification,
    "threat_detection": _aircraft_plan_threat_detection,
    "behavioral_anomaly": _aircraft_plan_behavioral_anomaly,
    "decision_support": _aircraft_plan_decision_support,
    "coordination": _aircraft_plan_coordination,
    "post_event_reporting": _aircraft_plan_post_event_reporting,
    "spatial_flexible": _aircraft_plan_spatial_flexible,
}


# ---------------------------------------------------------------------------
# LLM fallback for non-standard task types
# ---------------------------------------------------------------------------

_LLM_PLAN_SYSTEM = """\
You are generating a gold tool-call trace for a state-aware map agent task.
Given the blueprint and scenario, output a JSON array of tool calls.
Each element: {"tool": "<tool_name>", "args": {<args>}}.
Use $<tool_name>.<field> to reference results of previous calls.
Return JSON array only, no markdown fences."""


def _plan_via_llm(bp: dict, scenario: dict, llm: LLMBackend) -> list[dict]:
    prompt = (
        f"Blueprint:\n{json.dumps(bp, indent=2, ensure_ascii=False)}\n\n"
        f"Scenario ID: {scenario['scenario_id']}\n"
        f"Available object IDs: {json.dumps([o['id'] for cat in scenario['objects'].values() for o in cat])}\n\n"
        "Generate the gold tool trace as a JSON array."
    )
    try:
        result = llm.generate_json(prompt, system=_LLM_PLAN_SYSTEM, temperature=0.3)
        if isinstance(result, list):
            return result
        return result.get("trace", result.get("steps", []))
    except ValueError:
        logger.error("LLM gold plan generation failed for blueprint %s", bp.get("task_type"))
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_TEMPLATE_RE = re.compile(r"^\{\{\s*([^}|]+?)(?:\|([^}]+?))?\s*\}\}$")
_INLINE_TEMPLATE_RE = re.compile(r"\{\{\s*([^}|]+?)(?:\|([^}]+?))?\s*\}\}")


def generate_gold_plan(
    blueprint: dict,
    scenario: dict,
    llm: LLMBackend | None = None,
    *,
    context: PipelineContext | None = None,
) -> list[dict]:
    """Generate a gold tool-call trace from a blueprint.

    Uses deterministic templates where possible; falls back to LLM.
    """
    if blueprint.get("negative_case") or blueprint.get("should_ask_clarification"):
        return []

    task_type = blueprint.get("task_type", "")
    if context and task_type in context.profile.plan_templates:
        return _plan_from_template(context.profile.plan_templates[task_type], blueprint)

    planner = _DETERMINISTIC_PLANS.get(task_type) or _AIRCRAFT_DETERMINISTIC_PLANS.get(task_type)

    if planner is not None:
        return planner(blueprint, scenario)

    if llm is not None:
        return _plan_via_llm(blueprint, scenario, llm)

    logger.warning("No deterministic plan for %s and no LLM available", task_type)
    return []


def build_expected(
    blueprint: dict,
    gold_trace: list[dict],
    *,
    context: PipelineContext | None = None,
) -> dict:
    """Build the 'expected' section from the blueprint and gold trace."""
    allowed_extra_tools = (
        context.profile.allowed_extra_tools
        if context and context.profile.allowed_extra_tools
        else ["get_object_info", "resolve_entity", "get_current_map_state"]
    )
    return {
        "required_semantic_steps": blueprint.get("required_semantic_steps", []),
        "allowed_extra_tools": allowed_extra_tools,
        "forbidden_tools": [],
        "final_state_assertions": blueprint.get("expected_final_state", []),
        "should_ask_clarification": blueprint.get("should_ask_clarification", False),
    }


def _plan_from_template(template: list[dict[str, Any]], blueprint: dict) -> list[dict]:
    trace: list[dict] = []
    for raw_step in template:
        condition = raw_step.get("when")
        if condition and not _condition_matches(condition, blueprint):
            continue
        step = {
            "tool": raw_step["tool"],
            "args": _resolve_template_value(raw_step.get("args", {}), blueprint),
        }
        trace.append(step)
    return trace


def _condition_matches(condition: dict[str, Any], blueprint: dict) -> bool:
    actual = _get_path(blueprint, condition.get("path", ""))
    if "equals" in condition:
        return actual == condition["equals"]
    if "not_equals" in condition:
        return actual != condition["not_equals"]
    return bool(actual)


def _resolve_template_value(value: Any, blueprint: dict) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_template_value(v, blueprint) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_template_value(v, blueprint) for v in value]
    if not isinstance(value, str):
        return value

    full_match = _TEMPLATE_RE.match(value)
    if full_match:
        return _template_lookup(blueprint, full_match.group(1), full_match.group(2))

    def replace(match: re.Match) -> str:
        resolved = _template_lookup(blueprint, match.group(1), match.group(2))
        return "" if resolved is None else str(resolved)

    return _INLINE_TEMPLATE_RE.sub(replace, value)


def _template_lookup(blueprint: dict, path: str, default: str | None = None) -> Any:
    value = _get_path(blueprint, path.strip())
    if value is None and default is not None:
        return _coerce_default(default.strip())
    return value


def _get_path(data: dict, path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not part:
            continue
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _coerce_default(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
