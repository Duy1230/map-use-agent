from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from src.domain.profile import DomainProfile
from src.gold_plan_generator.generator import generate_gold_plan
from src.dedup.dedup import deduplicate
from src.mock_environment.database import ScenarioDatabase
from src.negative_generator.generator import generate_negative_cases
from src.pipeline.runner import _assemble_sample
from src.schemas import FinalStateAssertion, Scenario
from src.verification.execution_checker import check_execution
from src.pipeline.context import PipelineContext
from src.verification.assertion_checker import check_assertions
from src.verification.static_checker import check_static


def test_default_profile_loads_domain_vocabulary_and_tool_catalog() -> None:
    profile = DomainProfile.from_file(Path("profiles/traffic_map.yaml"))

    assert profile.name == "traffic_map"
    assert profile.selection_slots == ["object", "line", "polygon"]
    assert profile.layers == ["vehicles", "cameras", "roads", "zones"]
    assert profile.object_type("vehicle").collection == "vehicles"
    assert profile.time_ranges["last_10_minutes"] == 10
    assert "draw_vehicle_trajectory" in profile.task_family_names
    assert "get_vehicle_trajectory" in profile.tool_names
    assert "draw_vehicle_trajectory" in profile.plan_templates


def test_static_checker_uses_catalog_tool_names_and_profile_enums(tmp_path: Path) -> None:
    context = _custom_context(
        tmp_path,
        tools=[
            {
                "name": "custom_probe",
                "description": "Probe a configured asset.",
                "category": "object",
                "parameters": [
                    {"name": "object_id", "type": "string", "required": True},
                    {
                        "name": "time_range",
                        "type": "string",
                        "required": True,
                        "enum": ["custom_window"],
                    },
                ],
            }
        ],
        time_ranges={"custom_window": 7},
    )
    scenario = _scenario(
        objects_by_type={
            "asset": [
                {
                    "id": "asset_1",
                    "type": "asset",
                    "name": "Asset 1",
                    "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                }
            ]
        }
    )
    sample = {
        "id": "s1",
        "scenario_id": "scenario_001",
        "task_type": "custom_task",
        "user_request": "probe asset",
        "initial_state": {"selected": {"asset": None}},
        "expected": {},
        "gold_trace": [
            {
                "tool": "custom_probe",
                "args": {"object_id": "asset_1", "time_range": "custom_window"},
            }
        ],
    }

    ok, errors = check_static(sample, scenario, context=context)

    assert ok, errors


def test_scenario_database_indexes_legacy_and_generic_objects(tmp_path: Path) -> None:
    context = _custom_context(tmp_path)
    scenario = _scenario(
        legacy_objects={
            "vehicles": [
                {
                    "id": "vehicle_1",
                    "type": "vehicle",
                    "name": "Vehicle 1",
                    "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                }
            ]
        },
        objects_by_type={
            "asset": [
                {
                    "id": "asset_1",
                    "type": "asset",
                    "name": "Asset 1",
                    "aliases": ["main asset"],
                    "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
                }
            ]
        },
    )

    db = ScenarioDatabase(scenario, profile=context.profile)

    assert db.get_object("vehicle_1")["type"] == "vehicle"
    assert db.get_object("asset_1")["type"] == "asset"
    assert db.resolve_reference("main asset", "asset")["id"] == "asset_1"
    assert [obj["id"] for obj in db.get_objects_by_type("asset")] == ["asset_1"]


def test_declarative_plan_template_resolves_blueprint_paths(tmp_path: Path) -> None:
    context = _custom_context(
        tmp_path,
        plan_templates={
            "drop_custom_marker": [
                {
                    "tool": "draw_marker",
                    "args": {
                        "coordinates": "{{ target.coordinates }}",
                        "label": "{{ constraints.label }}",
                    },
                }
            ]
        },
    )
    blueprint = {
        "task_type": "drop_custom_marker",
        "target": {"coordinates": [106.7, 10.7]},
        "constraints": {"label": "Checkpoint"},
    }

    trace = generate_gold_plan(blueprint, _scenario(), context=context)

    assert trace == [
        {
            "tool": "draw_marker",
            "args": {"coordinates": [106.7, 10.7], "label": "Checkpoint"},
        }
    ]


def test_assertion_checker_rejects_unknown_assertions() -> None:
    context = PipelineContext.from_profile(
        DomainProfile.from_file(Path("profiles/traffic_map.yaml"))
    )

    ok, errors = check_assertions(
        {"drawn_artifacts": [], "highlighted_objects": [], "popups": []},
        [{"type": "not_registered"}],
        context=context,
    )

    assert not ok
    assert "Unknown assertion type" in errors[0]


def test_negative_generation_uses_profile_selection_slots_and_wrong_type_pairs(
    tmp_path: Path,
) -> None:
    context = _custom_context(
        tmp_path,
        object_types={
            "asset": {"collection": "assets", "selection_slot": "asset"},
            "sensor": {"collection": "sensors", "selection_slot": "sensor"},
        },
        selection_slots=["asset", "sensor"],
        wrong_type_pairs={"asset": "sensor"},
    )
    blueprint = {
        "task_type": "inspect_asset",
        "difficulty": "L2",
        "target": {
            "reference_mode": "selected_object",
            "expected_type": "asset",
            "object_id": "asset_1",
        },
        "initial_state": {
            "selected": {"asset": {"id": "asset_1", "type": "asset", "name": "Asset 1"}}
        },
        "expected_final_state": [{"type": "popup_shown", "target_id": "asset_1"}],
    }
    scenario = _scenario(
        objects_by_type={
            "asset": [_object("asset_1", "asset")],
            "sensor": [_object("sensor_1", "sensor")],
        }
    )

    negatives = generate_negative_cases(
        [blueprint],
        scenario,
        rng=random.Random(1),
        context=context,
    )

    mismatch = [case for case in negatives if case.get("_negative_type") == "type_mismatch"]
    assert mismatch
    assert mismatch[0]["initial_state"]["selected"] == {
        "asset": None,
        "sensor": {"id": "sensor_1", "type": "sensor", "name": "Sensor 1"},
    }


def test_assembly_and_execution_use_profile_defaults(tmp_path: Path) -> None:
    context = _custom_context(
        tmp_path,
        tools=[
            {
                "name": "draw_marker",
                "description": "Draw marker.",
                "category": "action",
                "parameters": [],
            }
        ],
        object_types={"asset": {"collection": "assets", "selection_slot": "asset"}},
        selection_slots=["asset"],
    )
    scenario = _scenario(
        objects_by_type={
            "asset": [
                {
                    "id": "asset_1",
                    "type": "asset",
                    "name": "Asset 1",
                    "geometry": {"type": "Point", "coordinates": [106.7, 10.7]},
                }
            ]
        }
    )
    blueprint = {
        "task_type": "drop_marker",
        "difficulty": "L1",
        "_gold_trace": [
            {
                "tool": "draw_marker",
                "args": {"coordinates": [106.7, 10.7], "label": "Asset"},
            }
        ],
        "_expected": {"final_state_assertions": [{"type": "artifact_exists"}]},
    }

    sample = _assemble_sample(blueprint, scenario, "mark asset", context=context)
    ok, errors, final_state = check_execution(sample, scenario, context=context)

    assert sample["tool_catalog_version"] == "custom_tools_v0.1"
    assert sample["initial_state"]["selected"] == {"asset": None}
    assert sample["initial_state"]["active_layers"] == ["assets"]
    assert ok, errors
    assert final_state is not None
    assert final_state["selected"] == {"asset": None}


def test_dedup_uses_profile_selection_slots(tmp_path: Path) -> None:
    context = _custom_context(
        tmp_path,
        object_types={"asset": {"collection": "assets", "selection_slot": "asset"}},
        selection_slots=["asset"],
    )
    samples = [
        _sample_for_dedup("s1", "inspect first", "asset_1"),
        _sample_for_dedup("s2", "inspect second", "asset_2"),
    ]

    kept = deduplicate(samples, context=context)

    assert [sample["id"] for sample in kept] == ["s1", "s2"]


def test_schemas_allow_profile_defined_assertions_and_generic_objects() -> None:
    assertion = FinalStateAssertion(type="custom_profile_assertion")
    scenario = Scenario(
        scenario_id="scenario_001",
        time="2026-05-18T09:15:00+07:00",
        objects={},
        objects_by_type={"asset": [_object("asset_1", "asset")]},
    )

    assert assertion.type == "custom_profile_assertion"
    assert scenario.objects_by_type["asset"][0].id == "asset_1"


def _custom_context(
    tmp_path: Path,
    *,
    tools: list[dict] | None = None,
    object_types: dict | None = None,
    selection_slots: list[str] | None = None,
    time_ranges: dict[str, int] | None = None,
    plan_templates: dict | None = None,
    wrong_type_pairs: dict[str, str] | None = None,
) -> PipelineContext:
    catalog_path = tmp_path / "tools.json"
    catalog_path.write_text(
        json.dumps(
            {
                "version": "custom_tools_v0.1",
                "tools": tools
                or [
                    {
                        "name": "draw_marker",
                        "description": "Draw marker.",
                        "category": "action",
                        "parameters": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "name": "custom",
                "tool_catalog": str(catalog_path.name),
                "object_types": object_types
                or {
                    "vehicle": {"collection": "vehicles", "selection_slot": "object"},
                    "asset": {"collection": "assets", "selection_slot": "asset"},
                },
                "selection_slots": selection_slots or ["object", "asset"],
                "layers": ["assets"],
                "time_ranges": time_ranges or {"last_10_minutes": 10},
                "task_families": [{"name": "custom_task", "difficulty": ["L1"]}],
                "difficulty_guidance": {"L1": "single step"},
                "assertions": ["artifact_exists", "popup_shown", "clarification_requested"],
                "negative_strategies": {
                    "enabled": [
                        "missing_selected",
                        "type_mismatch",
                        "multiple_selected",
                        "nonexistent_id",
                    ],
                    "wrong_type_pairs": wrong_type_pairs or {"vehicle": "asset"},
                },
                "plan_templates": plan_templates or {},
            }
        ),
        encoding="utf-8",
    )
    return PipelineContext.from_profile(DomainProfile.from_file(profile_path))


def _scenario(
    *,
    legacy_objects: dict | None = None,
    objects_by_type: dict | None = None,
) -> dict:
    return {
        "scenario_id": "scenario_001",
        "time": "2026-05-18T09:15:00+07:00",
        "objects": legacy_objects or {},
        "objects_by_type": objects_by_type or {},
        "time_series": {},
    }


def _object(object_id: str, object_type: str) -> dict:
    return {
        "id": object_id,
        "type": object_type,
        "name": object_id.replace("_", " ").title(),
        "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
    }


def _sample_for_dedup(sample_id: str, user_request: str, asset_id: str) -> dict:
    return {
        "id": sample_id,
        "scenario_id": "scenario_001",
        "task_type": "inspect_asset",
        "user_request": user_request,
        "initial_state": {
            "selected": {"asset": {"id": asset_id, "type": "asset", "name": asset_id}}
        },
        "gold_trace": [{"tool": "show_popup", "args": {"target_id": asset_id}}],
    }
