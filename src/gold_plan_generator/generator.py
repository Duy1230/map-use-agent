"""Gold plan generator — maps blueprints to canonical tool traces.

Uses deterministic mapping for well-known task types and falls back to LLM
for complex multi-step plans.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.llm_backend.base import LLMBackend

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

    steps.append({
        "tool": "get_vehicle_trajectory",
        "args": {"vehicle_id": vehicle_id, "time_range": time_range},
    })
    steps.append({
        "tool": "draw_polyline",
        "args": {
            "points": "$get_vehicle_trajectory.points",
            "label": f"Trajectory of {vehicle_id}",
            "source_object_id": vehicle_id,
        },
    })
    return steps


def _plan_show_object_info(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    object_id = target.get("object_id")
    steps: list[dict] = [{"tool": "get_current_map_state", "args": {}}]
    if target.get("reference_mode") == "selected_object":
        steps.append({"tool": "get_selected_entity", "args": {"entity_type": "object"}})
    steps.append({"tool": "get_object_info", "args": {"object_id": object_id}})
    steps.append({
        "tool": "show_popup",
        "args": {"target_id": object_id, "content": f"Info for {object_id}"},
    })
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
    steps.append({
        "tool": "get_objects_inside_polygon",
        "args": {"polygon_id": polygon_id, "object_type": obj_type},
    })
    steps.append({
        "tool": "highlight_objects",
        "args": {"object_ids": "$get_objects_inside_polygon.results"},
    })
    return steps


def _plan_objects_crossing_line(bp: dict, scenario: dict) -> list[dict]:
    target = bp.get("target", {})
    line_id = target.get("object_id")
    time_range = bp.get("constraints", {}).get("time_range", "last_10_minutes")
    obj_type = bp.get("constraints", {}).get("object_type", "vehicle")
    return [
        {"tool": "get_current_map_state", "args": {}},
        {"tool": "get_objects_crossing_line", "args": {
            "line_id": line_id, "object_type": obj_type, "time_range": time_range,
        }},
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
        {"tool": "get_nearby_objects", "args": {
            "object_id": object_id, "object_type": obj_type, "radius_meters": radius,
        }},
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

def generate_gold_plan(
    blueprint: dict,
    scenario: dict,
    llm: LLMBackend | None = None,
) -> list[dict]:
    """Generate a gold tool-call trace from a blueprint.

    Uses deterministic templates where possible; falls back to LLM.
    """
    if blueprint.get("negative_case") or blueprint.get("should_ask_clarification"):
        return []

    task_type = blueprint.get("task_type", "")
    planner = _DETERMINISTIC_PLANS.get(task_type)

    if planner is not None:
        return planner(blueprint, scenario)

    if llm is not None:
        return _plan_via_llm(blueprint, scenario, llm)

    logger.warning("No deterministic plan for %s and no LLM available", task_type)
    return []


def build_expected(
    blueprint: dict,
    gold_trace: list[dict],
) -> dict:
    """Build the 'expected' section from the blueprint and gold trace."""
    return {
        "required_semantic_steps": blueprint.get("required_semantic_steps", []),
        "allowed_extra_tools": ["get_object_info", "resolve_entity", "get_current_map_state"],
        "forbidden_tools": [],
        "final_state_assertions": blueprint.get("expected_final_state", []),
        "should_ask_clarification": blueprint.get("should_ask_clarification", False),
    }
