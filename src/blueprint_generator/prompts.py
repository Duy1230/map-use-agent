"""Prompt templates for LLM-driven blueprint generation."""

BLUEPRINT_SYSTEM = """\
You are generating synthetic evaluation blueprints for a state-aware map tool-use agent.

The agent has no vision. It can only use:
1. User request (natural language)
2. Structured map state (selected entities, active layers, drawn artifacts)
3. Available tools (canonical tool catalog)

Generate a task blueprint — NOT a natural language user request.

Rules:
- Do not invent object IDs outside the provided scenario.
- If the task uses "this vehicle", the initial state must contain a selected vehicle.
- If the task uses "this polygon", the initial state must contain a selected polygon.
- If the task uses "this line", the initial state must contain a selected line.
- For negative cases, explicitly specify why the agent must ask for clarification.
- Every task must have final_state_assertions.
- Prefer semantic correctness over linguistic diversity.
- Return valid JSON only, no markdown fences."""

BLUEPRINT_USER = """\
Scenario:
{scenario_json}

Available tools:
{tool_names}

Requested task family:
{task_family}

Difficulty:
{difficulty}

Generate a single task blueprint as a JSON object with these fields:
- task_type (string)
- difficulty (string, one of L0-L5)
- target (object with reference_mode, expected_type, object_id)
- constraints (object, may be empty)
- required_semantic_steps (array of strings)
- expected_final_state (array of assertion objects, each with at least "type")
- should_ask_clarification (boolean)
- negative_case (boolean)
- initial_state (object with selected.object, selected.line, selected.polygon, active_layers)

Return JSON only."""

# Task families to sample from
TASK_FAMILIES = [
    "draw_vehicle_trajectory",
    "show_object_info",
    "highlight_selected_entity",
    "objects_inside_polygon",
    "objects_crossing_line",
    "polygon_area",
    "line_length",
    "nearby_objects",
    "artifact_modification",
    "unsupported_or_ambiguous",
]

# Mapping from difficulty to what kind of blueprint to request
DIFFICULTY_GUIDANCE = {
    "L0": "Clarification or no-tool task. The user request is ambiguous, references something missing, or the agent should ask for more info.",
    "L1": "Single tool call with an explicit object ID reference.",
    "L2": "Selected reference resolution — the user says 'this vehicle' / 'this polygon' and the agent must resolve from state.",
    "L3": "Multi-step tool chain — at least 2-3 tool calls in sequence.",
    "L4": "Multi-turn or artifact-aware interaction — the task depends on previously drawn artifacts.",
    "L5": "Adversarial — ambiguity, type mismatch, visual reference, or dangerous action.",
}


# ---------------------------------------------------------------------------
# Aircraft track analysis domain prompts
# ---------------------------------------------------------------------------

AIRCRAFT_BLUEPRINT_SYSTEM = """\
You are generating synthetic evaluation blueprints for an aircraft track analysis agent.

The agent is a tool-use LLM serving military radar operators (VQ system). It can only use:
1. User request (natural language, often Vietnamese military jargon)
2. The 14-tool aircraft analysis catalog (getTrackInfo, getTrackPoints, getKinematicStats, analyzeSpatial, detectBehavior, lookupAircraftModel, lookupFlightPlan, findRelated, findNearestPOI, computeDistance, getReferenceData, mapDraw, mapControl, drawChart)
3. Scenario state with tracks, POIs, restricted zones, routes

Generate a task blueprint — NOT a natural language user request.

Rules:
- Do not invent track IDs outside the provided scenario.
- Every task must reference a valid trackId unless it is an adversarial case.
- For adversarial cases, specify why the agent must clarify or acknowledge scope limits.
- Tool arguments must use exact enum values from the catalog.
- Return valid JSON only, no markdown fences."""

AIRCRAFT_BLUEPRINT_USER = """\
Scenario:
{scenario_json}

Available tools:
{tool_names}

Requested task family:
{task_family}

Difficulty:
{difficulty}

Generate a single task blueprint as a JSON object with these fields:
- task_type (string, e.g. routine_surveillance, target_identification, threat_detection, behavioral_anomaly, decision_support, coordination, post_event_reporting, spatial_flexible, adversarial)
- difficulty (string: Easy, Medium, or Hard)
- target (object with reference_mode, expected_type, object_id — object_id is a trackId)
- constraints (object, may include pattern, dimension, aspect, poiType, etc.)
- required_semantic_steps (array of strings)
- expected_final_state (array of assertion objects with at least "type")
- should_ask_clarification (boolean)
- negative_case (boolean)
- initial_state (object, typically empty for aircraft domain)

Return JSON only."""

AIRCRAFT_TASK_FAMILIES = [
    "routine_surveillance",
    "target_identification",
    "threat_detection",
    "behavioral_anomaly",
    "decision_support",
    "coordination",
    "post_event_reporting",
    "spatial_flexible",
    "adversarial",
]

AIRCRAFT_DIFFICULTY_GUIDANCE = {
    "Easy": "Single tool call or simple lookup. Direct answer with minimal orchestration.",
    "Medium": "2-3 tool calls, conditional logic, or comparison between data sources.",
    "Hard": "4+ tool calls, multi-track aggregation, cross-reference, formatted output, or adversarial edge cases.",
}
