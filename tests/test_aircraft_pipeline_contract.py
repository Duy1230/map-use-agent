"""End-to-end contract tests for the aircraft profile.

These tests cover the boundaries that are easiest to break when adding a new
domain: catalog return shapes, gold-plan variable references, static checking,
execution evidence, and CLI world-generation dispatch.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from scripts.run_pipeline import app
from src.domain.profile import DomainProfile
from src.gold_plan_generator.generator import build_expected, generate_gold_plan
from src.pipeline.context import PipelineContext
from src.pipeline.runner import _assemble_sample
from src.verification.assertion_checker import check_assertions
from src.verification.execution_checker import check_execution
from src.verification.static_checker import check_static
from src.world_generator.aircraft_generator import generate_aircraft_world


def _context() -> PipelineContext:
    profile = DomainProfile.from_file(Path("profiles/aircraft_track.yaml"))
    return PipelineContext.from_profile(profile)


def _sample_for_blueprint(blueprint: dict, scenario: dict, context: PipelineContext) -> dict:
    trace = generate_gold_plan(blueprint, scenario, llm=None, context=context)
    blueprint["_gold_trace"] = trace
    blueprint["_expected"] = build_expected(blueprint, trace, context=context)
    return _assemble_sample(blueprint, scenario, "Review aircraft track.", context=context)


def test_aircraft_routine_plan_executes_and_draws_latest_point() -> None:
    context = _context()
    scenario = generate_aircraft_world("contract_routine", seed=42, num_tracks=1)
    track_id = scenario["objects"]["tracks"][0]["id"]
    blueprint = {
        "task_type": "routine_surveillance",
        "difficulty": "Easy",
        "target": {
            "reference_mode": "explicit_id",
            "expected_type": "track",
            "object_id": track_id,
        },
        "constraints": {},
        "required_semantic_steps": [],
        "expected_final_state": [{"type": "track_info_returned"}, {"type": "map_drawn"}],
        "initial_state": {},
    }

    sample = _sample_for_blueprint(blueprint, scenario, context)

    assert check_static(sample, scenario, context=context) == (True, [])
    ok, errors, final_state = check_execution(sample, scenario, context=context)
    assert ok, errors
    assert check_assertions(
        final_state,
        sample["expected"]["final_state_assertions"],
        context=context,
    ) == (True, [])


def test_aircraft_behavior_plan_forwards_findings_to_map_draw() -> None:
    context = _context()
    scenario = generate_aircraft_world("contract_behavior", seed=42, num_tracks=1)
    track = scenario["objects"]["tracks"][0]
    track["properties"]["behaviorTags"] = ["hovering"]
    blueprint = {
        "task_type": "behavioral_anomaly",
        "difficulty": "Medium",
        "target": {
            "reference_mode": "explicit_id",
            "expected_type": "track",
            "object_id": track["id"],
        },
        "constraints": {"pattern": "hovering"},
        "required_semantic_steps": [],
        "expected_final_state": [{"type": "behavior_detected"}, {"type": "map_drawn"}],
        "initial_state": {},
    }

    sample = _sample_for_blueprint(blueprint, scenario, context)

    ok, errors, final_state = check_execution(sample, scenario, context=context)
    assert ok, errors
    assert check_assertions(
        final_state,
        sample["expected"]["final_state_assertions"],
        context=context,
    ) == (True, [])


def test_aircraft_static_checker_allows_callsign_value_but_validates_track_values() -> None:
    context = _context()
    scenario = generate_aircraft_world("contract_static", seed=42, num_tracks=1)
    track = scenario["objects"]["tracks"][0]
    callsign_sample = {
        "initial_state": {"selected": {}},
        "gold_trace": [
            {
                "tool": "lookupAircraftModel",
                "args": {"by": "callsign", "value": track["callSign"]},
            }
        ],
        "expected": {"should_ask_clarification": False},
    }
    bad_track_value_sample = {
        "initial_state": {"selected": {}},
        "gold_trace": [
            {
                "tool": "lookupAircraftModel",
                "args": {"by": "kinematics", "value": "T-NOPE"},
            }
        ],
        "expected": {"should_ask_clarification": False},
    }

    assert check_static(callsign_sample, scenario, context=context) == (True, [])
    ok, errors = check_static(bad_track_value_sample, scenario, context=context)
    assert not ok
    assert any("T-NOPE" in error for error in errors)


def test_aircraft_assertions_require_execution_evidence() -> None:
    context = _context()

    ok, errors = check_assertions(
        {"tool_results": {}},
        [{"type": "track_info_returned"}, {"type": "behavior_detected"}],
        context=context,
    )

    assert not ok
    assert "track_info_returned" in errors[0]


def test_cli_worlds_stage_uses_aircraft_profile(tmp_path: Path, monkeypatch) -> None:
    profile_path = Path("profiles/aircraft_track.yaml").resolve()
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--stage",
            "worlds",
            "--num-worlds",
            "1",
            "--profile",
            str(profile_path),
        ],
    )

    assert result.exit_code == 0, result.output
    scenario_path = next((tmp_path / "scenarios").glob("*.json"))
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    assert "tracks" in scenario["objects"]
    assert "vehicles" not in scenario["objects"]
