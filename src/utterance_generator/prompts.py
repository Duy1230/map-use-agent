"""Prompt templates for LLM-driven user utterance generation."""

UTTERANCE_SYSTEM = """\
Generate natural user requests for a map-agent task blueprint.

Rules:
- Preserve the exact intent of the blueprint.
- Do not add new constraints beyond what the blueprint specifies.
- Do not mention object ID if reference_mode = selected_object. Use deictic expressions:
  "xe này", "vùng này", "đường này", "this vehicle", "this zone", "this road".
- Generate variants in multiple styles and languages.
- Return a JSON object with a single key "utterances" containing an array of strings.
- No markdown fences."""

UTTERANCE_USER = """\
Blueprint:
{blueprint_json}

Generate 6 user request variants in these styles:
1. Vietnamese formal
2. Vietnamese casual
3. English formal
4. Vietnamese-English code-mixed
5. Short command (≤8 words)
6. Natural conversational (longer, with filler words or context)

Return JSON only: {{"utterances": [...]}}"""


AIRCRAFT_UTTERANCE_SYSTEM = """\
Generate Vietnamese user requests only for an aircraft track analysis agent.

The speaker is a Vietnamese military radar/VQ operator. The request should
sound operational, concise, and mission-focused, not like a casual chatbot.

Rules:
- Preserve the exact intent of the blueprint.
- Keep the referenced TrackID, callsign, registration, enum values, and other identifiers exact.
- Use Vietnamese military/operational vocabulary naturally.
- Prefer terms such as mục tiêu, track, quỹ đạo, tọa độ, độ cao, tốc độ,
  hướng bay, vùng hạn chế, xâm nhập, mất tín hiệu, vẽ lên bản đồ, báo cáo nhanh.
- Do not use English phrasing except technical tokens such as TrackID,
  callsign, registration, radar, VQ, or exact enum values.
- Do not invent new constraints beyond the blueprint.
- Return a JSON object with a single key "utterances" containing an array of strings.
- No markdown fences."""


AIRCRAFT_UTTERANCE_USER = """\
Blueprint:
{blueprint_json}

Generate 6 Vietnamese user request variants, one for each configured style.
All variants must sound like they could be written by a Vietnamese military
radar/VQ operator.

Configured styles:
{style_guidance}

Profile rules:
{prompt_rules}

Return JSON only: {{"utterances": [...]}}"""
