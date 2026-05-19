"""Blueprint generator - uses LLM to create task blueprints from scenarios."""

from __future__ import annotations

import json
import logging
import random

from src.blueprint_generator.prompts import (
    AIRCRAFT_BLUEPRINT_SYSTEM,
    AIRCRAFT_BLUEPRINT_USER,
    BLUEPRINT_SYSTEM,
    BLUEPRINT_USER,
    DIFFICULTY_GUIDANCE,
    TASK_FAMILIES,
)
from src.pipeline.context import _AIRCRAFT_PROFILE_NAMES
from src.domain.scenario import compact_scenario_objects
from src.llm_backend.base import LLMBackend
from src.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


def _compact_scenario(
    scenario: dict,
    *,
    context: PipelineContext | None = None,
    per_type_limit: int = 5,
) -> str:
    """Create a compact scenario payload to stay within token limits."""
    compact = {
        "scenario_id": scenario["scenario_id"],
        "time": scenario["time"],
        "objects": compact_scenario_objects(
            scenario,
            context.profile if context else None,
            per_type_limit=per_type_limit,
        ),
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
    context: PipelineContext | None = None,
) -> dict | None:
    """Generate a single task blueprint via the LLM."""
    if tool_names is None:
        tool_names = context.tool_names if context else []

    is_aircraft = context and context.profile.name in _AIRCRAFT_PROFILE_NAMES
    difficulty_guidance = context.profile.difficulty_guidance if context else DIFFICULTY_GUIDANCE
    system_prompt = AIRCRAFT_BLUEPRINT_SYSTEM if is_aircraft else BLUEPRINT_SYSTEM
    user_template = AIRCRAFT_BLUEPRINT_USER if is_aircraft else BLUEPRINT_USER

    prompt = user_template.format(
        scenario_json=_compact_scenario(scenario, context=context),
        tool_names=json.dumps(tool_names),
        task_family=task_family,
        difficulty=f"{difficulty} - {difficulty_guidance.get(difficulty, '')}",
    )
    if context and context.profile.prompt_rules:
        prompt += "\nProfile rules:\n" + "\n".join(
            f"- {rule}" for rule in context.profile.prompt_rules
        )

    try:
        return llm.generate_json(prompt, system=system_prompt, temperature=temperature)
    except ValueError:
        logger.error("Failed to generate blueprint for %s/%s", task_family, difficulty)
        return None


def generate_blueprints_for_scenario(
    llm: LLMBackend,
    scenario: dict,
    *,
    target_count: int = 10,
    rng: random.Random | None = None,
    context: PipelineContext | None = None,
) -> list[dict]:
    """Generate multiple blueprints for one scenario."""
    r = rng or random.Random()
    difficulties = list(
        (context.profile.difficulty_guidance if context else DIFFICULTY_GUIDANCE).keys()
    )
    task_families = context.profile.task_family_names if context else TASK_FAMILIES

    blueprints: list[dict] = []
    for _ in range(target_count):
        family = r.choice(task_families)
        diff = r.choice(difficulties)
        bp = generate_blueprint(llm, scenario, family, diff, context=context)
        if bp is not None:
            bp.setdefault("_meta", {})
            bp["_meta"]["scenario_id"] = scenario["scenario_id"]
            bp["_meta"]["requested_family"] = family
            bp["_meta"]["requested_difficulty"] = diff
            blueprints.append(bp)

    return blueprints
