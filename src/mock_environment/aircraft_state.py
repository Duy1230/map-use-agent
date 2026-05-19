"""Mutable agent state for aircraft track analysis — tracks map layers, charts, and focus."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GeoFeatureRecord:
    feature_id: str
    layer_id: str
    feature_type: str
    geometry: dict | None = None
    properties: dict | None = None


@dataclass
class ChartRecord:
    chart_id: str
    chart_type: str
    title: str = ""
    data: dict | None = None


@dataclass
class AircraftAgentState:
    time: str = ""
    layers: dict[str, list[GeoFeatureRecord]] = field(default_factory=dict)
    charts: list[ChartRecord] = field(default_factory=list)
    focused_layer: str | None = None
    highlighted_time_range: dict | None = None
    tool_results: dict[str, list[Any]] = field(default_factory=dict)

    def map_draw(self, features: list[dict], options: dict | None = None) -> str:
        opts = options or {}
        layer_id = opts.get("layerId") or f"layer_{uuid.uuid4().hex[:8]}"
        records: list[GeoFeatureRecord] = []
        for feat in features:
            fid = f"feat_{uuid.uuid4().hex[:8]}"
            records.append(GeoFeatureRecord(
                feature_id=fid,
                layer_id=layer_id,
                feature_type=feat.get("type", "point"),
                geometry=feat.get("geometry"),
                properties=feat.get("properties"),
            ))
        self.layers.setdefault(layer_id, []).extend(records)
        if opts.get("autoFocus"):
            self.focused_layer = layer_id
        return layer_id

    def map_control(self, action: str, args: dict | None = None) -> dict:
        args = args or {}
        if action == "focus":
            self.focused_layer = args.get("layerId")
            return {"focused": self.focused_layer}
        if action == "clearLayer":
            lid = args.get("layerId")
            if lid and lid in self.layers:
                del self.layers[lid]
            return {"cleared": lid}
        if action == "highlightTimeRange":
            self.highlighted_time_range = args.get("timeRange")
            return {"highlighted": self.highlighted_time_range}
        if action == "listLayers":
            return {"layers": list(self.layers.keys())}
        return {}

    def draw_chart(self, chart_type: str, data: dict, title: str = "") -> str:
        cid = f"chart_{uuid.uuid4().hex[:8]}"
        self.charts.append(ChartRecord(
            chart_id=cid,
            chart_type=chart_type,
            title=title,
            data=data,
        ))
        return cid

    def record_tool_result(self, tool_name: str, result: Any) -> None:
        self.tool_results.setdefault(tool_name, []).append(result)

    def visible_to_agent(self) -> dict:
        return {
            "time": self.time,
            "layers": {
                lid: [
                    {"feature_id": r.feature_id, "type": r.feature_type, "properties": r.properties}
                    for r in records
                ]
                for lid, records in self.layers.items()
            },
            "charts": [
                {"chart_id": c.chart_id, "type": c.chart_type, "title": c.title}
                for c in self.charts
            ],
            "focused_layer": self.focused_layer,
            "tool_results": self.tool_results,
        }
