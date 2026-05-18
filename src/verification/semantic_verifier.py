"""Gate 5 — LLM-based semantic verification."""

from __future__ import annotations

import json
import logging

from src.llm_backend.base import LLMBackend

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM = """\
You are a strict verifier for a synthetic map tool-use dataset.

The agent has no vision and must not infer objects from visual descriptions.
It can only use structured state and tools.

Check whether the candidate sample is semantically valid.

Evaluate:
1. Does the user request match the task_type and blueprint intent?
2. Is the target object resolvable from the initial state or explicit ID/name?
3. Are the required semantic steps correct for the task?
4. Are final_state_assertions sufficient to verify correctness?
5. Are there any hallucinated object IDs, geometries, tools, or constraints?
6. For negative cases, should the agent ask clarification instead of calling tools?

Return JSON only:
{"valid": true/false, "issues": [...], "suggested_fix": null or string}"""

VERIFIER_USER = """\
Candidate sample:
{candidate_json}"""


def verify_semantic(
    llm: LLMBackend,
    sample: dict,
    *,
    temperature: float = 0.2,
) -> tuple[bool, list[str]]:
    """Run LLM semantic verification on a candidate sample.

    Returns (passed, list_of_issues).
    """
    compact = {
        k: sample.get(k)
        for k in (
            "id",
            "task_type",
            "difficulty",
            "user_request",
            "initial_state",
            "gold_trace",
            "expected",
            "tags",
        )
    }

    prompt = VERIFIER_USER.format(candidate_json=json.dumps(compact, indent=2, ensure_ascii=False))

    try:
        result = llm.generate_json(prompt, system=VERIFIER_SYSTEM, temperature=temperature)
    except ValueError:
        logger.error("Semantic verification failed for %s", sample.get("id"))
        return (False, ["LLM verifier returned invalid JSON"])

    is_valid = result.get("valid", False)
    issues = result.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]

    return (is_valid, issues)
