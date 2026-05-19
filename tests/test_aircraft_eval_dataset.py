"""Validate the 50 eval test cases from the aircraft agent eval dataset.

Ensures each test case parses correctly, references tools from the catalog,
and has the expected structural fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain.profile import DomainProfile

_EVAL_DATASET_PATH = Path(".external/aircraft_agent_eval_dataset_v1.json")


def _load_eval_dataset() -> list[dict]:
    if not _EVAL_DATASET_PATH.exists():
        pytest.skip(f"Optional internal eval dataset not found: {_EVAL_DATASET_PATH}")
    data = json.loads(_EVAL_DATASET_PATH.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("test_cases", data.get("samples", []))


def _load_aircraft_tool_names() -> set[str]:
    profile = DomainProfile.from_file(Path("profiles/aircraft_track.yaml"))
    return set(profile.tool_names)


def test_eval_dataset_loads() -> None:
    cases = _load_eval_dataset()
    assert len(cases) == 50, f"Expected 50 test cases, got {len(cases)}"


def test_eval_dataset_structure() -> None:
    cases = _load_eval_dataset()
    required_fields = {"prompt", "expected_tools", "expected_answer_includes"}

    for i, case in enumerate(cases):
        for field in required_fields:
            assert field in case, f"Test case {case.get('id', i)} missing field {field!r}"


def test_eval_dataset_tool_references_valid() -> None:
    cases = _load_eval_dataset()
    valid_tools = _load_aircraft_tool_names()

    for i, case in enumerate(cases):
        for tool_call in case.get("expected_tools", []):
            tool_name = tool_call if isinstance(tool_call, str) else tool_call.get("tool", "")
            assert tool_name in valid_tools, (
                f"Test case {case.get('id', i)} references unknown tool {tool_name!r}. "
                f"Valid tools: {sorted(valid_tools)}"
            )


def test_eval_dataset_has_domain_and_difficulty() -> None:
    cases = _load_eval_dataset()
    valid_difficulties = {"Easy", "Medium", "Hard"}

    for i, case in enumerate(cases):
        assert "domain" in case, f"Test case {case.get('id', i)} missing domain"
        difficulty = case.get("difficulty", "")
        assert difficulty in valid_difficulties, (
            f"Test case {case.get('id', i)} has unexpected difficulty {difficulty!r}"
        )


def test_eval_dataset_has_trajectory_types() -> None:
    cases = _load_eval_dataset()

    traj_types = {case.get("trajectory_type", "") for case in cases}
    traj_types.discard("")
    assert len(traj_types) > 1, "Expected multiple trajectory types in the dataset"


def test_eval_dataset_ids_are_unique() -> None:
    cases = _load_eval_dataset()

    ids = [case.get("id", "") for case in cases]
    assert len(ids) == len(set(ids)), "Eval dataset has duplicate IDs"
