"""Mutable map state that tracks what the agent can see and what has changed."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class DrawnArtifact:
    artifact_id: str
    artifact_type: str
    label: str | None = None
    source_object_id: str | None = None
    geometry: dict | None = None


@dataclass
class Popup:
    popup_id: str
    target_id: str
    content: str


@dataclass
class MapState:
    time: str = ""
    selected_object: dict | None = None
    selected_line: dict | None = None
    selected_polygon: dict | None = None
    active_layers: list[str] = field(default_factory=lambda: ["vehicles", "cameras", "roads", "zones"])
    drawn_artifacts: list[DrawnArtifact] = field(default_factory=list)
    highlighted_objects: set[str] = field(default_factory=set)
    popups: list[Popup] = field(default_factory=list)

    # --- mutations --------------------------------------------------------

    def add_artifact(
        self,
        artifact_type: str,
        geometry: dict | None = None,
        label: str | None = None,
        source_object_id: str | None = None,
    ) -> str:
        aid = f"artifact_{uuid.uuid4().hex[:8]}"
        self.drawn_artifacts.append(
            DrawnArtifact(
                artifact_id=aid,
                artifact_type=artifact_type,
                label=label,
                source_object_id=source_object_id,
                geometry=geometry,
            )
        )
        return aid

    def add_popup(self, target_id: str, content: str) -> str:
        pid = f"popup_{uuid.uuid4().hex[:8]}"
        self.popups.append(Popup(popup_id=pid, target_id=target_id, content=content))
        return pid

    def clear(self, scope: str) -> None:
        if scope == "all":
            self.drawn_artifacts.clear()
            self.highlighted_objects.clear()
            self.popups.clear()
        elif scope == "polylines":
            self.drawn_artifacts = [a for a in self.drawn_artifacts if a.artifact_type != "polyline"]
        elif scope == "polygons":
            self.drawn_artifacts = [a for a in self.drawn_artifacts if a.artifact_type != "polygon"]
        elif scope == "markers":
            self.drawn_artifacts = [a for a in self.drawn_artifacts if a.artifact_type != "marker"]
        elif scope == "popups":
            self.popups.clear()
        elif scope == "highlights":
            self.highlighted_objects.clear()

    # --- serialisation for agent ------------------------------------------

    def visible_to_agent(self) -> dict:
        """Return the subset of state visible to the agent."""
        return {
            "time": self.time,
            "selected": {
                "object": self.selected_object,
                "line": self.selected_line,
                "polygon": self.selected_polygon,
            },
            "active_layers": self.active_layers,
            "drawn_artifacts": [
                {
                    "artifact_id": a.artifact_id,
                    "artifact_type": a.artifact_type,
                    "label": a.label,
                    "source_object_id": a.source_object_id,
                }
                for a in self.drawn_artifacts
            ],
            "highlighted_objects": sorted(self.highlighted_objects),
            "popups": [
                {"popup_id": p.popup_id, "target_id": p.target_id, "content": p.content}
                for p in self.popups
            ],
        }
