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

    if atype in ("clarification_requested", "no_tool_called"):
        return True

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
