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
