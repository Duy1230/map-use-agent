"""Scenario access helpers that support legacy and flexible object layouts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.domain.profile import DomainProfile


def iter_scenario_objects(
    scenario: dict[str, Any],
    profile: DomainProfile | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield objects from legacy collections and generic objects_by_type."""
    seen: set[str] = set()

    legacy = scenario.get("objects", {}) or {}
    for collection in legacy.values():
        if isinstance(collection, list):
            for obj in collection:
                oid = obj.get("id")
                if oid and oid not in seen:
                    seen.add(oid)
                    yield obj

    by_type = scenario.get("objects_by_type", {}) or {}
    for collection in by_type.values():
        if isinstance(collection, list):
            for obj in collection:
                oid = obj.get("id")
                if oid and oid not in seen:
                    seen.add(oid)
                    yield obj

    if profile is None:
        return

    for definition in profile.object_types.values():
        collection = legacy.get(definition.collection, [])
        if isinstance(collection, list):
            for obj in collection:
                oid = obj.get("id")
                if oid and oid not in seen:
                    seen.add(oid)
                    yield obj


def scenario_objects_by_type(
    scenario: dict[str, Any],
    profile: DomainProfile | None = None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for obj in iter_scenario_objects(scenario, profile):
        grouped.setdefault(obj.get("type", ""), []).append(obj)
    return grouped


def scenario_objects_for_type(
    scenario: dict[str, Any],
    object_type: str,
    profile: DomainProfile | None = None,
) -> list[dict[str, Any]]:
    return scenario_objects_by_type(scenario, profile).get(object_type, [])


def compact_scenario_objects(
    scenario: dict[str, Any],
    profile: DomainProfile | None = None,
    per_type_limit: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    grouped = scenario_objects_by_type(scenario, profile)
    compact: dict[str, list[dict[str, Any]]] = {}
    for object_type, objects in grouped.items():
        collection = profile.collection_for_type(object_type) if profile else f"{object_type}s"
        compact[collection] = objects[:per_type_limit]
    return compact
