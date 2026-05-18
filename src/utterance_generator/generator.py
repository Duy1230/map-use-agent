"""Utterance generator — produces multi-style user requests from validated blueprints."""

from __future__ import annotations

import json
import logging

from src.llm_backend.base import LLMBackend
from src.pipeline.context import PipelineContext
from src.utterance_generator.prompts import UTTERANCE_SYSTEM, UTTERANCE_USER

logger = logging.getLogger(__name__)


def generate_utterances(
    llm: LLMBackend,
    blueprint: dict,
    *,
    temperature: float = 0.9,
    context: PipelineContext | None = None,
) -> list[str]:
    """Generate a list of user utterance variants for a single blueprint.

    Returns an empty list if the LLM fails after retries.
    """
    bp_compact = {
        k: v
        for k, v in blueprint.items()
        if k
        in (
            "task_type",
            "difficulty",
            "target",
            "constraints",
            "should_ask_clarification",
            "negative_case",
        )
    }

    prompt = UTTERANCE_USER.format(
        blueprint_json=json.dumps(bp_compact, indent=2, ensure_ascii=False)
    )
    if context and context.profile.utterance_styles:
        prompt += "\nConfigured utterance styles:\n" + "\n".join(
            f"- {style}" for style in context.profile.utterance_styles
        )
    if context and context.profile.prompt_rules:
        prompt += "\nProfile rules:\n" + "\n".join(
            f"- {rule}" for rule in context.profile.prompt_rules
        )

    try:
        data = llm.generate_json(prompt, system=UTTERANCE_SYSTEM, temperature=temperature)
    except ValueError:
        logger.error("Utterance generation failed for blueprint %s", blueprint.get("task_type"))
        return []

    utterances = data.get("utterances", [])
    if not isinstance(utterances, list):
        return []

    return [u for u in utterances if isinstance(u, str) and u.strip()]
