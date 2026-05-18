"""Gate 1 — JSON Schema validation for candidate samples."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import jsonschema

logger = logging.getLogger(__name__)

_SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent / "schemas"
_TASK_SCHEMA: dict | None = None


def _load_task_schema() -> dict:
    global _TASK_SCHEMA
    if _TASK_SCHEMA is None:
        path = _SCHEMA_DIR / "task.schema.json"
        _TASK_SCHEMA = json.loads(path.read_text(encoding="utf-8"))
    return _TASK_SCHEMA


def validate_schema(sample: dict) -> tuple[bool, list[str]]:
    """Validate a candidate sample against the task JSON schema.

    Returns (passed, list_of_error_messages).
    """
    required_fields = [
        "id",
        "scenario_id",
        "task_type",
        "user_request",
        "initial_state",
        "expected",
    ]
    errors: list[str] = []

    for field in required_fields:
        if field not in sample:
            errors.append(f"Missing required field: {field!r}")

    initial = sample.get("initial_state", {})
    if not isinstance(initial, dict):
        errors.append("initial_state must be a dict")
    else:
        if "selected" not in initial:
            errors.append("initial_state missing 'selected'")

    expected = sample.get("expected", {})
    if not isinstance(expected, dict):
        errors.append("expected must be a dict")

    gold_trace = sample.get("gold_trace", [])
    if not isinstance(gold_trace, list):
        errors.append("gold_trace must be a list")
    else:
        for i, step in enumerate(gold_trace):
            if not isinstance(step, dict) or "tool" not in step:
                errors.append(f"gold_trace[{i}] must be a dict with 'tool' key")

    # Attempt full JSON Schema validation if schema file exists
    try:
        schema = _load_task_schema()
        jsonschema.validate(sample, schema)
    except jsonschema.ValidationError as exc:
        errors.append(f"JSON Schema: {exc.message}")
    except FileNotFoundError:
        pass  # schema not exported yet — skip

    return (len(errors) == 0, errors)
