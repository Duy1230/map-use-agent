"""MockMapEnvironment — wires state, database, and tools into an executable sandbox."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.domain.profile import DomainProfile
from src.mock_environment.database import ObjectNotFoundError, ScenarioDatabase, TypeMismatchError
from src.mock_environment.state import MapState
from src.mock_environment.tools import TOOL_REGISTRY, ToolExecutionError
from src.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

_VAR_RE = re.compile(r"^\$(\w+)\.(\w+)$")


class MockMapEnvironment:
    """Deterministic execution sandbox for canonical tool traces."""

    def __init__(
        self,
        scenario: dict,
        initial_state_override: dict | None = None,
        *,
        context: PipelineContext | None = None,
        profile: DomainProfile | None = None,
    ) -> None:
        self.context = context
        self.profile = profile or (context.profile if context else None)
        self.db = ScenarioDatabase(scenario, profile=self.profile)
        self.state = MapState(time=scenario["time"])
        if self.profile:
            self.state.selected_entities = {slot: None for slot in self.profile.selection_slots}
            self.state.active_layers = list(self.profile.layers)

        if initial_state_override:
            sel = initial_state_override.get("selected", {})
            if self.profile:
                for slot in self.profile.selection_slots:
                    self.state.selected_entities[slot] = sel.get(slot)
            else:
                self.state.selected_object = sel.get("object")
                self.state.selected_line = sel.get("line")
                self.state.selected_polygon = sel.get("polygon")
            if "active_layers" in initial_state_override:
                self.state.active_layers = initial_state_override["active_layers"]

        self.execution_log: list[dict] = []

    # ------------------------------------------------------------------

    def execute_tool(self, name: str, args: dict) -> dict:
        """Execute a single canonical tool call. Raises on failure."""
        registry = self.context.tool_registry if self.context else TOOL_REGISTRY
        if name not in registry:
            raise ToolExecutionError(f"Unknown tool: {name!r}")

        resolved_args = self._resolve_vars(args)

        try:
            result = registry[name](self.state, self.db, resolved_args)
        except (ObjectNotFoundError, TypeMismatchError) as exc:
            raise ToolExecutionError(str(exc)) from exc

        self.execution_log.append(
            {
                "tool": name,
                "args": resolved_args,
                "result": result,
            }
        )
        return result

    def execute_trace(self, trace: list[dict]) -> dict:
        """Execute a full gold trace and return the final visible state."""
        for step in trace:
            self.execute_tool(step["tool"], step.get("args", {}))
        return self.get_final_state()

    def get_final_state(self) -> dict:
        return self.state.visible_to_agent()

    # ------------------------------------------------------------------

    def _resolve_vars(self, args: dict) -> dict:
        """Replace $tool_name.field references with values from execution_log."""
        resolved: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str):
                m = _VAR_RE.match(value)
                if m:
                    tool_name, field = m.group(1), m.group(2)
                    resolved[key] = self._lookup_result(tool_name, field)
                    continue
            resolved[key] = value
        return resolved

    def _lookup_result(self, tool_name: str, field: str) -> Any:
        for entry in reversed(self.execution_log):
            if entry["tool"] == tool_name:
                result = entry["result"]
                if field in result:
                    return result[field]
                raise ToolExecutionError(f"Field {field!r} not found in result of {tool_name}")
        raise ToolExecutionError(f"No previous execution of tool {tool_name!r}")
