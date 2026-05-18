from __future__ import annotations

import logging

from openai import OpenAI

from src.llm_backend.base import LLMBackend, LLMResponse

logger = logging.getLogger(__name__)


class APIBackend(LLMBackend):
    """Generic OpenAI / OpenAI-compatible API backend."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.model = model
        kwargs: dict = {"timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self._client = OpenAI(**kwargs)

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        completion = self._client.chat.completions.create(**kwargs)

        choice = completion.choices[0]
        usage = {}
        if completion.usage:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }

        return LLMResponse(
            text=choice.message.content or "",
            usage=usage,
            model=completion.model,
            raw=completion.model_dump(),
        )
