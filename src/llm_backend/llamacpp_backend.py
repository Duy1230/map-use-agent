from __future__ import annotations

import logging

import httpx

from src.llm_backend.base import LLMBackend, LLMResponse

logger = logging.getLogger(__name__)


class LlamaCppBackend(LLMBackend):
    """Connects to a llama.cpp server (OpenAI-compatible or native endpoint)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "local-model",
        *,
        api_key: str = "no-key",
        timeout: float = 120.0,
        use_openai_compat: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.use_openai_compat = use_openai_compat
        self._client = httpx.Client(timeout=timeout)

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        if self.use_openai_compat:
            return self._generate_openai_compat(
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        return self._generate_native(prompt, temperature=temperature, max_tokens=max_tokens)

    def _generate_openai_compat(
        self,
        prompt: str,
        *,
        system: str,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        return LLMResponse(
            text=choice["message"]["content"],
            usage=data.get("usage", {}),
            model=data.get("model", self.model),
            raw=data,
        )

    def _generate_native(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "n_predict": max_tokens,
        }
        resp = self._client.post(f"{self.base_url}/completion", json=payload)
        resp.raise_for_status()
        data = resp.json()

        return LLMResponse(
            text=data.get("content", ""),
            usage={
                "prompt_tokens": data.get("tokens_evaluated", 0),
                "completion_tokens": data.get("tokens_predicted", 0),
            },
            model=self.model,
            raw=data,
        )
