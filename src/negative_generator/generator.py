"""Negative / adversarial case generator.

Combines deterministic state mutations with LLM-generated adversarial utterances.
"""

from __future__ import annotations

import copy
import json
import logging
import random
from typing import Any

from src.llm_backend.base import LLMBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic mutation strategies
# ---------------------------------------------------------------------------

def _mutate_missing_selected(blueprint: dict) -> dict | None:
    """Remove the selected entity so the agent has no target to resolve."""
    target = blueprint.get("target", {})
    if target.get("reference_mode") != "selected_object":
        return None

    neg = copy.deepcopy(blueprint)
    neg["negative_case"] = True
    neg["should_ask_clarification"] = True
    neg["difficulty"] = "L0"

    neg.setdefault("initial_state", {})
    neg["initial_state"]["selected"] = {
        "object": None, "line": None, "polygon": None
    }
    neg["expected_final_state"] = [
        {"type": "clarification_requested"}
    ]
    neg["_negative_type"] = "missing_selected"
    return neg


def _mutate_type_mismatch(blueprint: dict, scenario: dict) -> dict | None:
    """Replace the selected entity with one of the wrong type."""
    target = blueprint.get("target", {})
    expected_type = target.get("expected_type")
    if not expected_type or target.get("reference_mode") != "selected_object":
        return None

    wrong_type = {"vehicle": "camera", "camera": "vehicle", "polygon": "line", "line": "polygon"}
    replacement_type = wrong_type.get(expected_type)
    if not replacement_type:
        return None

    candidates = scenario.get("objects", {}).get(f"{replacement_type}s", [])
    if not candidates:
        return None

    wrong_obj = candidates[0]
    neg = copy.deepcopy(blueprint)
    neg["negative_case"] = True
    neg["should_ask_clarification"] = True
    neg["difficulty"] = "L5"

    slot = "object" if replacement_type in ("vehicle", "camera") else replacement_type
    neg.setdefault("initial_state", {})
    neg["initial_state"]["selected"] = {
        "object": None, "line": None, "polygon": None,
        slot: {"id": wrong_obj["id"], "type": wrong_obj["type"], "name": wrong_obj["name"]},
    }
    neg["expected_final_state"] = [
        {"type": "clarification_requested"}
    ]
    neg["_negative_type"] = "type_mismatch"
    return neg


def _mutate_multiple_selected(blueprint: dict, scenario: dict) -> dict | None:
    """Put multiple entities in the selected slot to force clarification."""
    target = blueprint.get("target", {})
    expected_type = target.get("expected_type")
    if not expected_type:
        return None

    category = f"{expected_type}s"
    candidates = scenario.get("objects", {}).get(category, [])
    if len(candidates) < 2:
        return None

    neg = copy.deepcopy(blueprint)
    neg["negative_case"] = True
    neg["should_ask_clarification"] = True
    neg["difficulty"] = "L5"

    objs = [
        {"id": c["id"], "type": c["type"], "name": c["name"]}
        for c in candidates[:2]
    ]
    slot = "object" if expected_type in ("vehicle", "camera") else expected_type
    neg.setdefault("initial_state", {})
    neg["initial_state"]["selected"] = {
        "object": None, "line": None, "polygon": None,
        slot: objs,
    }
    neg["expected_final_state"] = [
        {"type": "clarification_requested"}
    ]
    neg["_negative_type"] = "multiple_selected"
    return neg


def _mutate_nonexistent_id(blueprint: dict) -> dict | None:
    """Replace the target object_id with a fake one."""
    target = blueprint.get("target", {})
    if target.get("reference_mode") != "explicit_id":
        return None

    neg = copy.deepcopy(blueprint)
    neg["negative_case"] = True
    neg["should_ask_clarification"] = True
    neg["difficulty"] = "L5"
    neg["target"]["object_id"] = "vehicle_999_nonexistent"
    neg["expected_final_state"] = [
        {"type": "clarification_requested"}
    ]
    neg["_negative_type"] = "nonexistent_id"
    return neg


DETERMINISTIC_MUTATORS = [
    _mutate_missing_selected,
    _mutate_type_mismatch,
    _mutate_multiple_selected,
    _mutate_nonexistent_id,
]


# ---------------------------------------------------------------------------
# LLM-generated adversarial cases
# ---------------------------------------------------------------------------

_ADVERSARIAL_SYSTEM = """\
You are generating adversarial/negative test cases for a map tool-use agent.
The agent has no vision. It must ask for clarification when it cannot resolve
the user's intent from structured state alone.

Generate a single negative-case blueprint. Return JSON only."""

_ADVERSARIAL_USER = """\
Scenario ID: {scenario_id}
Available object IDs: {object_ids}

Generate one of these adversarial patterns:
- Visual reference: user refers to "the red car on the left" (agent has no vision)
- Unsupported action: user asks to delete an object or change the basemap
- Dangerous action: user asks to clear all data without confirmation
- Tool confusion: user asks for trajectory of a camera (cameras don't have trajectories)

Return a JSON blueprint with:
  task_type, difficulty (L5), target, negative_case: true,
  should_ask_clarification: true, expected_final_state,
  initial_state, _negative_type, user_hint (short description of the adversarial pattern)"""


def generate_llm_adversarial(
    llm: LLMBackend,
    scenario: dict,
) -> dict | None:
    """Use the LLM to generate one adversarial blueprint."""
    object_ids = [
        obj["id"]
        for cat in scenario.get("objects", {}).values()
        for obj in cat
    ]
    prompt = _ADVERSARIAL_USER.format(
        scenario_id=scenario["scenario_id"],
        object_ids=json.dumps(object_ids[:15]),
    )
    try:
        return llm.generate_json(prompt, system=_ADVERSARIAL_SYSTEM, temperature=0.9)
    except ValueError:
        logger.error("LLM adversarial generation failed")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_negative_cases(
    blueprints: list[dict],
    scenario: dict,
    *,
    llm: LLMBackend | None = None,
    llm_adversarial_count: int = 3,
    rng: random.Random | None = None,
) -> list[dict]:
    """Generate negative/adversarial variants from existing positive blueprints.

    Returns a list of negative-case blueprints.
    """
    r = rng or random.Random()
    negatives: list[dict] = []

    for bp in blueprints:
        if bp.get("negative_case"):
            continue
        for mutator in DETERMINISTIC_MUTATORS:
            if mutator.__code__.co_varnames[1:2] == ("scenario",):
                result = mutator(bp, scenario)
            else:
                result = mutator(bp)
            if result is not None:
                result.setdefault("_meta", {}).update(bp.get("_meta", {}))
                negatives.append(result)

    if llm is not None:
        for _ in range(llm_adversarial_count):
            adv = generate_llm_adversarial(llm, scenario)
            if adv is not None:
                adv.setdefault("_meta", {})["scenario_id"] = scenario["scenario_id"]
                negatives.append(adv)

    return negatives
