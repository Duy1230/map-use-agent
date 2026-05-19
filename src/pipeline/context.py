"""Runtime context shared across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.domain.profile import DomainProfile
from src.mock_environment.tools import TOOL_REGISTRY

_AIRCRAFT_PROFILE_NAMES = {"aircraft_track", "aircraft"}


def _select_tool_registry(profile: DomainProfile) -> dict:
    """Return the tool registry matching the profile domain."""
    if profile.name in _AIRCRAFT_PROFILE_NAMES:
        from src.mock_environment.aircraft_tools import AIRCRAFT_TOOL_REGISTRY

        return AIRCRAFT_TOOL_REGISTRY
    return TOOL_REGISTRY


@dataclass(frozen=True)
class PipelineContext:
    """Resolved runtime wiring for a pipeline run."""

    profile: DomainProfile
    tool_registry: dict

    @classmethod
    def from_profile(cls, profile: DomainProfile) -> PipelineContext:
        return cls(profile=profile, tool_registry=_select_tool_registry(profile))

    @classmethod
    def from_profile_path(cls, profile_path: str | Path) -> PipelineContext:
        return cls.from_profile(DomainProfile.from_file(profile_path))

    @property
    def tool_names(self) -> list[str]:
        return self.profile.tool_names

    @property
    def time_ranges(self) -> dict[str, int]:
        return self.profile.time_ranges

    @property
    def selection_slots(self) -> list[str]:
        return self.profile.selection_slots

    @property
    def active_layers(self) -> list[str]:
        return self.profile.layers

    def validate_tool_registry(self) -> list[str]:
        """Return missing executable handlers for catalog tools.

        Some catalog entries may intentionally be metadata-only in future profiles,
        but the default profile should be fully executable.
        """
        return [name for name in self.tool_names if name not in self.tool_registry]
