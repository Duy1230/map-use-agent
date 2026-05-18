from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    usage: dict = field(default_factory=dict)
    model: str = ""
    raw: dict | None = None

    def parse_json(self) -> dict:
        """Extract the first JSON object/array from the response text."""
        text = self.text.strip()
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try to find JSON within markdown fences
        for marker in ("```json", "```"):
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start)
                return json.loads(text[start:end].strip())
        raise ValueError(f"No valid JSON found in LLM response: {text[:200]}")


class LLMBackend(ABC):
    """Abstract interface for all LLM backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse: ...

    def generate_json(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        retries: int = 3,
    ) -> dict:
        """Generate and parse a JSON response, retrying on parse failures."""
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            resp = self.generate(
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            try:
                return resp.parse_json()
            except (ValueError, json.JSONDecodeError) as exc:
                last_err = exc
                logger.warning("JSON parse failed (attempt %d/%d): %s", attempt, retries, exc)
        raise ValueError(f"Failed to get valid JSON after {retries} attempts") from last_err

    def generate_batch(self, prompts: list[str], **kwargs) -> list[LLMResponse]:
        """Default sequential batch; backends may override with true batching."""
        return [self.generate(p, **kwargs) for p in prompts]
