"""Pydantic models — single source of truth for every data structure in the pipeline.

JSON Schema files in schemas/ are generated from these models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class GeoJSONPoint(BaseModel):
    type: str = "Point"
    coordinates: list[float] = Field(..., min_length=2, max_length=3)


class GeoJSONLineString(BaseModel):
    type: str = "LineString"
    coordinates: list[list[float]]


class GeoJSONPolygon(BaseModel):
    type: str = "Polygon"
    coordinates: list[list[list[float]]]


Geometry = GeoJSONPoint | GeoJSONLineString | GeoJSONPolygon


# ---------------------------------------------------------------------------
# Aircraft domain types
# ---------------------------------------------------------------------------


class AircraftTrackPoint(BaseModel):
    timestamp: str
    lat: float
    lon: float
    alt: float = 0.0
    speed: float = 0.0
    heading: float = 0.0
    climbRate: float = 0.0


class AircraftTrackIdentity(BaseModel):
    callSign: str = ""
    registration: str = ""
    type: str = ""
    aircraftModel: str = ""


class AircraftGeoFeature(BaseModel):
    type: str = "point"
    geometry: dict[str, Any] = Field(default_factory=dict)
    properties: dict[str, Any] = Field(default_factory=dict)


class Finding(BaseModel):
    description: str = ""
    severity: str = "info"
    timeRange: dict[str, Any] = Field(default_factory=dict)
    geoFeatures: list[AircraftGeoFeature] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class POI(BaseModel):
    id: str
    name: str
    kind: str
    point: dict[str, float] = Field(default_factory=dict)
    distanceKm: float | None = None


class AircraftLayerRecord(BaseModel):
    feature_id: str
    feature_type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class AircraftChartRecord(BaseModel):
    chart_id: str
    chart_type: str
    title: str = ""


class AircraftAgentStateSchema(BaseModel):
    """Schema for the aircraft agent's visible state."""

    time: str = ""
    layers: dict[str, list[AircraftLayerRecord]] = Field(default_factory=dict)
    charts: list[AircraftChartRecord] = Field(default_factory=list)
    focused_layer: str | None = None


# ---------------------------------------------------------------------------
# Scenario / World objects
# ---------------------------------------------------------------------------


class ObjectProperties(BaseModel):
    speed_kmh: float | None = None
    status: str | None = None
    heading_deg: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class MapObject(BaseModel):
    id: str
    type: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    geometry: GeoJSONPoint | GeoJSONPolygon | GeoJSONLineString
    properties: ObjectProperties = Field(default_factory=ObjectProperties)


class TrajectoryPoint(BaseModel):
    timestamp: str
    coordinates: list[float] = Field(..., min_length=2, max_length=3)


class ScenarioObjects(BaseModel):
    vehicles: list[MapObject] = Field(default_factory=list)
    cameras: list[MapObject] = Field(default_factory=list)
    polygons: list[MapObject] = Field(default_factory=list)
    lines: list[MapObject] = Field(default_factory=list)


class Scenario(BaseModel):
    scenario_id: str
    coordinate_system: str = "EPSG:4326"
    time: str
    objects: ScenarioObjects
    objects_by_type: dict[str, list[MapObject]] = Field(default_factory=dict)
    time_series: dict[str, dict[str, list[TrajectoryPoint]]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Map State (what the agent sees)
# ---------------------------------------------------------------------------


class SelectedEntity(BaseModel):
    id: str
    type: str
    name: str | None = None


class DrawnArtifact(BaseModel):
    artifact_id: str
    artifact_type: str
    label: str | None = None
    source_object_id: str | None = None
    geometry: dict | None = None


class MapState(BaseModel):
    time: str
    selected: dict[str, SelectedEntity | list[SelectedEntity] | None] = Field(
        default_factory=lambda: {"object": None, "line": None, "polygon": None}
    )
    active_layers: list[str] = Field(default_factory=list)
    drawn_artifacts: list[DrawnArtifact] = Field(default_factory=list)
    highlighted_objects: list[str] = Field(default_factory=list)
    popups: list[dict[str, str]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool Catalog
# ---------------------------------------------------------------------------


class ToolParameter(BaseModel):
    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    category: str
    parameters: list[ToolParameter] = Field(default_factory=list)
    return_schema: dict[str, Any] = Field(default_factory=dict)
    side_effects: bool = False


class ToolCatalog(BaseModel):
    version: str
    tools: list[ToolDefinition]


# ---------------------------------------------------------------------------
# Tool Call / Gold Trace
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------


class Difficulty(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"
    Easy = "Easy"
    Medium = "Medium"
    Hard = "Hard"


class BlueprintTarget(BaseModel):
    reference_mode: str
    expected_type: str | None = None
    object_id: str | None = None


class Blueprint(BaseModel):
    task_type: str
    difficulty: Difficulty
    target: BlueprintTarget
    constraints: dict[str, Any] = Field(default_factory=dict)
    required_semantic_steps: list[str]
    expected_final_state: list[dict[str, Any]]
    should_ask_clarification: bool = False
    negative_case: bool = False


# ---------------------------------------------------------------------------
# Final State Assertions
# ---------------------------------------------------------------------------


class AssertionType(str, Enum):
    # Traffic map assertions
    artifact_exists = "artifact_exists"
    object_highlighted = "object_highlighted"
    objects_highlighted = "objects_highlighted"
    popup_shown = "popup_shown"
    artifact_removed = "artifact_removed"
    polyline_source_matches_vehicle = "polyline_source_matches_vehicle"
    objects_inside_polygon_highlighted = "objects_inside_polygon_highlighted"
    no_tool_called = "no_tool_called"
    clarification_requested = "clarification_requested"
    # Aircraft track assertions
    track_info_returned = "track_info_returned"
    behavior_detected = "behavior_detected"
    no_behavior_detected = "no_behavior_detected"
    spatial_finding_returned = "spatial_finding_returned"
    map_drawn = "map_drawn"
    chart_drawn = "chart_drawn"
    distance_computed = "distance_computed"
    poi_found = "poi_found"
    related_found = "related_found"
    aircraft_model_identified = "aircraft_model_identified"
    flight_plan_returned = "flight_plan_returned"
    reference_data_returned = "reference_data_returned"
    scope_acknowledged = "scope_acknowledged"


class FinalStateAssertion(BaseModel):
    type: str
    artifact_type: str | None = None
    source_object_id: str | None = None
    object_id: str | None = None
    object_ids: list[str] | None = None
    object_type: str | None = None
    target_id: str | None = None
    time_range: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Expected behavior
# ---------------------------------------------------------------------------


class Expected(BaseModel):
    required_semantic_steps: list[str] = Field(default_factory=list)
    allowed_extra_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    final_state_assertions: list[FinalStateAssertion] = Field(default_factory=list)
    should_ask_clarification: bool = False


# ---------------------------------------------------------------------------
# Generation metadata
# ---------------------------------------------------------------------------


class GenerationMetadata(BaseModel):
    generator_model: str = ""
    blueprint_id: str = ""
    verified_by_schema: bool = False
    verified_by_execution: bool = False
    verified_by_llm: bool = False
    human_reviewed: bool = False


# ---------------------------------------------------------------------------
# Full Task Sample (single-turn)
# ---------------------------------------------------------------------------


class TaskSample(BaseModel):
    id: str
    scenario_id: str
    tool_catalog_version: str = "map_tools_v0.1"
    language: str = "vi"
    difficulty: Difficulty
    task_type: str
    user_request: str
    initial_state: MapState
    available_tools: list[str] = Field(default_factory=list)
    gold_trace: list[ToolCall] = Field(default_factory=list)
    expected: Expected
    tags: list[str] = Field(default_factory=list)
    generation_metadata: GenerationMetadata = Field(default_factory=GenerationMetadata)


# ---------------------------------------------------------------------------
# Multi-turn sample
# ---------------------------------------------------------------------------


class TurnSample(BaseModel):
    user: str
    expected_steps: list[str] = Field(default_factory=list)
    gold_trace: list[ToolCall] = Field(default_factory=list)


class MultiTurnSample(BaseModel):
    id: str
    scenario_id: str
    tool_catalog_version: str = "map_tools_v0.1"
    turns: list[TurnSample]
    initial_state: MapState
    final_state_assertions: list[FinalStateAssertion] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    generation_metadata: GenerationMetadata = Field(default_factory=GenerationMetadata)
