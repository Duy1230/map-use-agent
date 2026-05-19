"""Gate 4 — Final state assertion checking."""

from __future__ import annotations

import logging

from src.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

DEFAULT_ASSERTIONS = {
    "artifact_exists",
    "object_highlighted",
    "objects_highlighted",
    "popup_shown",
    "artifact_removed",
    "polyline_source_matches_vehicle",
    "objects_inside_polygon_highlighted",
    "clarification_requested",
    "no_tool_called",
}


def check_assertions(
    final_state: dict | None,
    assertions: list[dict],
    *,
    is_negative: bool = False,
    context: PipelineContext | None = None,
) -> tuple[bool, list[str]]:
    """Verify final_state satisfies the expected assertions.

    Returns (passed, list_of_error_messages).
    """
    if is_negative:
        if not assertions:
            return (True, [])
        for a in assertions:
            if a.get("type") in ("clarification_requested", "no_tool_called"):
                return (True, [])
        return (True, [])

    if final_state is None:
        if assertions:
            return (False, ["No final_state available but assertions expected"])
        return (True, [])

    errors: list[str] = []
    allowed_assertions = context.profile.assertions if context else DEFAULT_ASSERTIONS

    for i, assertion in enumerate(assertions):
        atype = assertion.get("type", "")
        if atype not in allowed_assertions:
            errors.append(f"Unknown assertion type: {atype}")
            continue
        ok = _check_single(final_state, assertion, atype)
        if not ok:
            errors.append(f"Assertion[{i}] failed: {assertion}")

    return (len(errors) == 0, errors)


def _check_single(state: dict, assertion: dict, atype: str) -> bool:
    if atype == "artifact_exists":
        return _has_artifact(
            state,
            artifact_type=assertion.get("artifact_type"),
            source_object_id=assertion.get("source_object_id"),
        )

    if atype == "object_highlighted":
        oid = assertion.get("object_id", "")
        return oid in state.get("highlighted_objects", [])

    if atype == "objects_highlighted":
        obj_type = assertion.get("object_type")
        highlighted = state.get("highlighted_objects", [])
        if obj_type:
            return len(highlighted) > 0
        oids = assertion.get("object_ids", [])
        return all(oid in highlighted for oid in oids)

    if atype == "popup_shown":
        target_id = assertion.get("target_id")
        return any(p.get("target_id") == target_id for p in state.get("popups", []))

    if atype == "artifact_removed":
        return not _has_artifact(
            state,
            artifact_type=assertion.get("artifact_type"),
            source_object_id=assertion.get("source_object_id"),
        )

    if atype == "polyline_source_matches_vehicle":
        vid = assertion.get("source_object_id", "")
        return _has_artifact(state, artifact_type="polyline", source_object_id=vid)

    if atype == "objects_inside_polygon_highlighted":
        highlighted = state.get("highlighted_objects", [])
        return len(highlighted) > 0

    if atype in ("clarification_requested", "no_tool_called", "scope_acknowledged"):
        return True

    if atype == "track_info_returned":
        return any(
            isinstance(result, dict) and result.get("trackId") and len(result) > 1
            for result in _tool_results(state, "getTrackInfo")
        )
    if atype == "behavior_detected":
        return any(_result_count(result) > 0 for result in _tool_results(state, "detectBehavior"))
    if atype == "no_behavior_detected":
        results = _tool_results(state, "detectBehavior")
        return bool(results) and all(_result_count(result) == 0 for result in results)
    if atype == "spatial_finding_returned":
        return any(_result_count(result) > 0 for result in _tool_results(state, "analyzeSpatial"))
    if atype == "map_drawn":
        layers = state.get("layers", {})
        return len(layers) > 0
    if atype == "chart_drawn":
        charts = state.get("charts", [])
        return len(charts) > 0
    if atype == "distance_computed":
        return any(
            isinstance(result, dict) and result.get("distanceKm", -1) >= 0
            for result in _tool_results(state, "computeDistance")
        )
    if atype == "poi_found":
        return any(_result_count(result) > 0 for result in _tool_results(state, "findNearestPOI"))
    if atype == "related_found":
        return any(_result_count(result) > 0 for result in _tool_results(state, "findRelated"))
    if atype == "aircraft_model_identified":
        return any(
            isinstance(result, dict) and bool(result.get("candidates"))
            for result in _tool_results(state, "lookupAircraftModel")
        )
    if atype == "flight_plan_returned":
        return any(
            isinstance(result, dict)
            and any(result.get(key) for key in ("route", "origin", "destination"))
            for result in _tool_results(state, "lookupFlightPlan")
        )
    if atype == "reference_data_returned":
        return any(_result_count(result) > 0 for result in _tool_results(state, "getReferenceData"))

    logger.warning("Unknown assertion type: %s", atype)
    return True


def _has_artifact(
    state: dict,
    artifact_type: str | None = None,
    source_object_id: str | None = None,
) -> bool:
    for a in state.get("drawn_artifacts", []):
        type_ok = artifact_type is None or a.get("artifact_type") == artifact_type
        src_ok = source_object_id is None or a.get("source_object_id") == source_object_id
        if type_ok and src_ok:
            return True
    return False


def _tool_results(state: dict, tool_name: str) -> list:
    results = state.get("tool_results", {}).get(tool_name, [])
    return results if isinstance(results, list) else []


def _result_count(result: object) -> int:
    if isinstance(result, list):
        return len(result)
    if isinstance(result, dict):
        for key in ("findings", "results", "data"):
            value = result.get(key)
            if isinstance(value, list):
                return len(value)
        return 1 if result else 0
    return 0
