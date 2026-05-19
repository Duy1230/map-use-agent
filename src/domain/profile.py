"""Domain profile loading.

A profile moves domain vocabulary and pipeline wiring out of Python constants.
The default traffic profile preserves the original map-agent behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ObjectTypeDef:
    """Profile definition for one object type."""

    name: str
    collection: str
    geometry: str | None = None
    selection_slot: str = "object"
    aliases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TaskFamilyDef:
    """Configurable task-family entry used by blueprint generation."""

    name: str
    difficulty: list[str] = field(default_factory=list)
    description: str = ""


@dataclass(frozen=True)
class ToolCatalogRuntime:
    """Small runtime wrapper around the JSON tool catalog."""

    version: str
    tools: list[dict[str, Any]]
    path: Path

    @classmethod
    def from_file(cls, path: str | Path) -> ToolCatalogRuntime:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            version=data.get("version", ""),
            tools=list(data.get("tools", [])),
            path=p,
        )

    @property
    def names(self) -> list[str]:
        return [tool.get("name", "") for tool in self.tools if tool.get("name")]

    def tool(self, name: str) -> dict[str, Any] | None:
        for tool in self.tools:
            if tool.get("name") == name:
                return tool
        return None

    def parameter(self, tool_name: str, parameter_name: str) -> dict[str, Any] | None:
        tool = self.tool(tool_name)
        if not tool:
            return None
        for parameter in tool.get("parameters", []):
            if parameter.get("name") == parameter_name:
                return parameter
        return None


@dataclass(frozen=True)
class DomainProfile:
    """Declarative domain profile used by the pipeline at runtime."""

    name: str
    path: Path
    tool_catalog: ToolCatalogRuntime
    object_types: dict[str, ObjectTypeDef]
    selection_slots: list[str]
    layers: list[str]
    time_ranges: dict[str, int]
    task_families: list[TaskFamilyDef]
    difficulty_guidance: dict[str, str]
    assertions: set[str]
    negative_strategies: dict[str, Any] = field(default_factory=dict)
    plan_templates: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    prompt_rules: list[str] = field(default_factory=list)
    utterance_styles: list[str] = field(default_factory=list)
    utterance_style_guidance: dict[str, str] = field(default_factory=dict)
    allowed_extra_tools: list[str] = field(default_factory=list)
    tool_type_constraints: dict[str, dict[str, str]] = field(default_factory=dict)
    object_reference_args: list[str] = field(default_factory=list)
    conditional_object_reference_args: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> DomainProfile:
        profile_path = Path(path)
        data = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        base = profile_path.parent

        catalog_path = Path(data.get("tool_catalog", "tool_catalogs/map_tools_v0_1.json"))
        if not catalog_path.is_absolute():
            catalog_path = (base / catalog_path).resolve()

        object_types = {
            name: ObjectTypeDef(
                name=name,
                collection=cfg.get("collection", f"{name}s"),
                geometry=cfg.get("geometry"),
                selection_slot=cfg.get("selection_slot", "object"),
                aliases=list(cfg.get("aliases", [])),
            )
            for name, cfg in data.get("object_types", {}).items()
        }
        task_families = [
            TaskFamilyDef(
                name=entry.get("name", ""),
                difficulty=list(entry.get("difficulty", [])),
                description=entry.get("description", ""),
            )
            for entry in data.get("task_families", [])
            if entry.get("name")
        ]

        return cls(
            name=data.get("name", profile_path.stem),
            path=profile_path,
            tool_catalog=ToolCatalogRuntime.from_file(catalog_path),
            object_types=object_types,
            selection_slots=list(data.get("selection_slots", [])),
            layers=list(data.get("layers", [])),
            time_ranges=dict(data.get("time_ranges", {})),
            task_families=task_families,
            difficulty_guidance=dict(data.get("difficulty_guidance", {})),
            assertions=set(data.get("assertions", [])),
            negative_strategies=dict(data.get("negative_strategies", {})),
            plan_templates=dict(data.get("plan_templates", {})),
            prompt_rules=list(data.get("prompt_rules", [])),
            utterance_styles=list(data.get("utterance_styles", [])),
            utterance_style_guidance=dict(data.get("utterance_style_guidance", {})),
            allowed_extra_tools=list(data.get("allowed_extra_tools", [])),
            tool_type_constraints=dict(data.get("tool_type_constraints", {})),
            object_reference_args=list(data.get("object_reference_args", [])),
            conditional_object_reference_args=dict(
                data.get("conditional_object_reference_args", {})
            ),
        )

    @property
    def tool_names(self) -> list[str]:
        return self.tool_catalog.names

    @property
    def task_family_names(self) -> list[str]:
        return [family.name for family in self.task_families]

    @property
    def enabled_negative_strategies(self) -> list[str]:
        return list(self.negative_strategies.get("enabled", []))

    @property
    def wrong_type_pairs(self) -> dict[str, str]:
        return dict(self.negative_strategies.get("wrong_type_pairs", {}))

    @property
    def adversarial_patterns(self) -> list[str]:
        return list(self.negative_strategies.get("adversarial_patterns", []))

    def object_type(self, name: str) -> ObjectTypeDef:
        try:
            return self.object_types[name]
        except KeyError as exc:
            raise KeyError(f"Unknown object type in profile {self.name!r}: {name!r}") from exc

    def collection_for_type(self, object_type: str) -> str:
        if object_type in self.object_types:
            return self.object_types[object_type].collection
        return f"{object_type}s"

    def type_for_collection(self, collection: str) -> str | None:
        for object_type, definition in self.object_types.items():
            if definition.collection == collection:
                return object_type
        return None

    def selection_slot_for_type(self, object_type: str) -> str:
        if object_type in self.object_types:
            return self.object_types[object_type].selection_slot
        if object_type in self.selection_slots:
            return object_type
        return "object"

    def parameter_enum(self, tool_name: str, parameter_name: str) -> list[Any] | None:
        parameter = self.tool_catalog.parameter(tool_name, parameter_name)
        enum = parameter.get("enum") if parameter else None
        if enum is not None:
            return list(enum)
        return None
