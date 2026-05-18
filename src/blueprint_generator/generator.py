"""Blueprint generator — uses LLM to create task blueprints from scenarios."""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from src.blueprint_generator.prompts import (
    BLUEPRINT_SYSTEM,
    BLUEPRINT_USER,
    DIFFICULTY_GUIDANCE,
    TASK_FAMILIES,
)
from src.llm_backend.base import LLMBackend

logger = logging.getLogger(__name__)


def _compact_scenario(scenario: dict, max_vehicles: int = 5) -> str:
    """Create a compact version of the scenario to stay within token limits."""
    compact = {
        "scenario_id": scenario["scenario_id"],
        "time": scenario["time"],
        "objects": {
            "vehicles": scenario["objects"]["vehicles"][:max_vehicles],
            "cameras": scenario["objects"]["cameras"][:5],
            "polygons": scenario["objects"]["polygons"][:3],
            "lines": scenario["objects"]["lines"][:3],
        },
    }
    return json.dumps(compact, indent=2, ensure_ascii=False)


def generate_blueprint(
    llm: LLMBackend,
    scenario: dict,
    task_family: str,
    difficulty: str,
    *,
    tool_names: list[str] | None = None,
    temperature: float = 0.8,
) -> dict | None:
    """Generate a single task blueprint via the LLM.

    Returns the parsed blueprint dict, or None if all retries fail.
    """
    if tool_names is None:
        tool_names = [
            "get_current_map_state", "get_selected_entity", "resolve_entity",
            "get_object_info", "get_vehicle_trajectory", "get_nearby_objects",
            "get_polygon_area", "get_line_length", "get_objects_inside_polygon",
            "get_objects_crossing_line", "highlight_objects", "draw_polyline",
            "draw_polygon", "draw_marker", "show_popup", "clear_artifacts",
        ]

    prompt = BLUEPRINT_USER.format(
        scenario_json=_compact_scenario(scenario),
        tool_names=json.dumps(tool_names),
        task_family=task_family,
        difficulty=f"{difficulty} — {DIFFICULTY_GUIDANCE.get(difficulty, '')}",
    )

    try:
        return llm.generate_json(
            prompt, system=BLUEPRINT_SYSTEM, temperature=temperature
        )
    except ValueError:
        logger.error("Failed to generate blueprint for %s/%s", task_family, difficulty)
        return None


def generate_blueprints_for_scenario(
    llm: LLMBackend,
    scenario: dict,
    *,
    target_count: int = 10,
    rng: random.Random | None = None,
) -> list[dict]:
    """Generate multiple blueprints for one scenario, sampling across families and difficulties."""
    r = rng or random.Random()
    difficulties = list(DIFFICULTY_GUIDANCE.keys())

    blueprints: list[dict] = []
    for _ in range(target_count):
        family = r.choice(TASK_FAMILIES)
        diff = r.choice(difficulties)
        bp = generate_blueprint(llm, scenario, family, diff)
        if bp is not None:
            bp.setdefault("_meta", {})
            bp["_meta"]["scenario_id"] = scenario["scenario_id"]
            bp["_meta"]["requested_family"] = family
            bp["_meta"]["requested_difficulty"] = diff
            blueprints.append(bp)

    return blueprints
