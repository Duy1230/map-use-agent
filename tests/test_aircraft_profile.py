"""Tests for the aircraft_track domain profile and tool catalog."""

from __future__ import annotations

from pathlib import Path

from src.domain.profile import DomainProfile
from src.pipeline.context import PipelineContext


def test_aircraft_profile_loads_successfully() -> None:
    profile = DomainProfile.from_file(Path("profiles/aircraft_track.yaml"))

    assert profile.name == "aircraft_track"
    assert "track" in profile.object_types
    assert "airport" in profile.object_types
    assert "base" in profile.object_types
    assert "restricted_zone" in profile.object_types
    assert "common_route" in profile.object_types


def test_aircraft_profile_has_correct_selection_slots() -> None:
    profile = DomainProfile.from_file(Path("profiles/aircraft_track.yaml"))

    assert "track" in profile.selection_slots
    assert "poi" in profile.selection_slots
    assert "zone" in profile.selection_slots


def test_aircraft_tool_catalog_has_14_tools() -> None:
    profile = DomainProfile.from_file(Path("profiles/aircraft_track.yaml"))

    assert len(profile.tool_names) == 14
    expected_tools = {
        "getTrackInfo", "getTrackPoints", "getKinematicStats",
        "analyzeSpatial", "detectBehavior", "lookupAircraftModel",
        "lookupFlightPlan", "findRelated", "findNearestPOI",
        "computeDistance", "getReferenceData", "mapDraw",
        "mapControl", "drawChart",
    }
    assert set(profile.tool_names) == expected_tools


def test_aircraft_task_families() -> None:
    profile = DomainProfile.from_file(Path("profiles/aircraft_track.yaml"))

    expected_families = {
        "routine_surveillance", "target_identification", "threat_detection",
        "behavioral_anomaly", "decision_support", "coordination",
        "post_event_reporting", "spatial_flexible", "adversarial",
    }
    assert set(profile.task_family_names) == expected_families


def test_aircraft_difficulty_guidance() -> None:
    profile = DomainProfile.from_file(Path("profiles/aircraft_track.yaml"))

    assert "Easy" in profile.difficulty_guidance
    assert "Medium" in profile.difficulty_guidance
    assert "Hard" in profile.difficulty_guidance


def test_aircraft_assertions_are_configured() -> None:
    profile = DomainProfile.from_file(Path("profiles/aircraft_track.yaml"))

    assert "track_info_returned" in profile.assertions
    assert "behavior_detected" in profile.assertions
    assert "map_drawn" in profile.assertions
    assert "chart_drawn" in profile.assertions
    assert "clarification_requested" in profile.assertions


def test_aircraft_context_selects_aircraft_tool_registry() -> None:
    profile = DomainProfile.from_file(Path("profiles/aircraft_track.yaml"))
    context = PipelineContext.from_profile(profile)

    assert "getTrackInfo" in context.tool_registry
    assert "detectBehavior" in context.tool_registry
    assert "mapDraw" in context.tool_registry
    assert len(context.tool_registry) == 14


def test_aircraft_negative_strategies() -> None:
    profile = DomainProfile.from_file(Path("profiles/aircraft_track.yaml"))

    enabled = profile.enabled_negative_strategies
    assert "ambiguous_reference" in enabled
    assert "out_of_scope" in enabled
    assert "missing_track_id" in enabled


def test_aircraft_type_constraints() -> None:
    profile = DomainProfile.from_file(Path("profiles/aircraft_track.yaml"))

    constraints = profile.tool_type_constraints
    assert constraints.get("getTrackInfo", {}).get("trackId") == "track"
    assert constraints.get("detectBehavior", {}).get("trackId") == "track"
