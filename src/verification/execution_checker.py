"""Gate 3 — Execute gold trace in the mock environment."""

from __future__ import annotations

import logging

from src.mock_environment.environment import MockMapEnvironment
from src.mock_environment.tools import ToolExecutionError
from src.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


def check_execution(
    sample: dict,
    scenario: dict,
    *,
    context: PipelineContext | None = None,
) -> tuple[bool, list[str], dict | None]:
    """Run the gold trace in the mock environment.

    Returns (passed, errors, final_state_or_None).
    """
    gold_trace = sample.get("gold_trace", [])

    if sample.get("expected", {}).get("should_ask_clarification", False):
        if len(gold_trace) == 0:
            return (True, [], None)

    if not gold_trace:
        return (True, [], None)

    errors: list[str] = []
    initial_state = sample.get("initial_state")

    try:
        env = MockMapEnvironment(scenario, initial_state, context=context)
    except Exception as exc:
        errors.append(f"Environment init failed: {exc}")
        return (False, errors, None)

    for i, step in enumerate(gold_trace):
        tool = step.get("tool", "")
        args = step.get("args", {})
        try:
            env.execute_tool(tool, args)
        except ToolExecutionError as exc:
            errors.append(f"gold_trace[{i}] ({tool}): {exc}")
        except Exception as exc:
            err_type = type(exc).__name__
            if "NotFound" in err_type or "ExecutionError" in err_type:
                errors.append(f"gold_trace[{i}] ({tool}): {exc}")
            else:
                errors.append(f"gold_trace[{i}] ({tool}) unexpected error: {exc}")

    final_state = env.get_final_state() if not errors else None
    return (len(errors) == 0, errors, final_state)
