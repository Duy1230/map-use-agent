from __future__ import annotations

import logging

from openai import OpenAI

from src.llm_backend.base import LLMBackend, LLMResponse

logger = logging.getLogger(__name__)


class VLLMBackend(LLMBackend):
    """Connects to a vLLM server via its OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "meta-llama/Llama-3.1-70B-Instruct",
        *,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        retry_backoff_max_seconds: float = 30.0,
        reconnect_on_failure: bool = True,
    ):
        self.model = model
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

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
