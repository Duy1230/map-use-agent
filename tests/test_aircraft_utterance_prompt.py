"""Tests for Vietnamese military-style aircraft utterance generation."""

from __future__ import annotations

from pathlib import Path

from src.domain.profile import DomainProfile
from src.llm_backend.base import LLMBackend, LLMResponse
from src.pipeline.context import PipelineContext
from src.utterance_generator.generator import generate_utterances


def test_aircraft_profile_prefers_vietnamese_military_utterance_styles() -> None:
    profile = DomainProfile.from_file(Path("profiles/aircraft_track.yaml"))

    assert "vi_military_command" in profile.utterance_styles
    assert "vi_shift_briefing" in profile.utterance_styles
    assert "en" not in profile.utterance_styles
    assert "vi_casual" not in profile.utterance_styles
    assert "vi_military_command" in profile.utterance_style_guidance


def test_aircraft_utterance_prompt_uses_vietnamese_military_voice() -> None:
    context = PipelineContext.from_profile(
        DomainProfile.from_file(Path("profiles/aircraft_track.yaml"))
    )
    llm = CapturingLLM()
    blueprint = {
        "task_type": "threat_detection",
        "difficulty": "Hard",
        "target": {
            "reference_mode": "explicit_id",
            "expected_type": "track",
            "object_id": "T-1742",
        },
        "constraints": {"aspect": "restrictedZones"},
        "should_ask_clarification": False,
        "negative_case": False,
    }

    utterances = generate_utterances(llm, blueprint, context=context)

    assert utterances == ["Báo cáo nhanh mục tiêu T-1742 có xâm nhập vùng hạn chế không."]
    assert "Vietnamese user requests only" in llm.system
    assert "Vietnamese military radar/VQ operator" in llm.system
    assert "vi_military_command" in llm.prompt
    assert "mục tiêu" in llm.prompt
    assert "vùng hạn chế" in llm.prompt
    assert "English formal" not in llm.prompt
    assert "Vietnamese-English code-mixed" not in llm.prompt


class CapturingLLM(LLMBackend):
    def __init__(self) -> None:
        self.prompt = ""
        self.system = ""

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        self.prompt = prompt
        self.system = system
        return LLMResponse(
            text='{"utterances": ["Báo cáo nhanh mục tiêu T-1742 có xâm nhập vùng hạn chế không."]}'
        )
