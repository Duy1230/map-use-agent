from __future__ import annotations

import logging
import time
from collections.abc import Callable

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
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        retry_backoff_max_seconds: float = 30.0,
        reconnect_on_failure: bool = True,
        client_factory: Callable[[float], httpx.Client] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.use_openai_compat = use_openai_compat
        self.max_retries = max(1, max_retries)
        self.retry_backoff_seconds = retry_backoff_seconds
        self.retry_backoff_max_seconds = retry_backoff_max_seconds
        self.reconnect_on_failure = reconnect_on_failure
        self._client_factory = client_factory or (lambda timeout: httpx.Client(timeout=timeout))
        self._client = self._client_factory(timeout)

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
        resp = self._post_with_retry(
            f"{self.base_url}/v1/chat/completions",
            json_payload=payload,
            headers=headers,
        )
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
        resp = self._post_with_retry(f"{self.base_url}/completion", json_payload=payload)
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

    def _post_with_retry(
        self,
        url: str,
        *,
        json_payload: dict,
        headers: dict | None = None,
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.post(url, json=json_payload, headers=headers)
                resp.raise_for_status()
                return resp
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.RemoteProtocolError,
                httpx.TimeoutException,
                httpx.HTTPStatusError,
            ) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                    raise
                last_exc = exc
                if attempt >= self.max_retries:
                    logger.error(
                        "llama.cpp request failed after %d attempts: %s",
                        self.max_retries,
                        exc,
                    )
                    raise
                sleep_s = min(
                    self.retry_backoff_seconds * (2 ** (attempt - 1)),
                    self.retry_backoff_max_seconds,
                )
                logger.warning(
                    "llama.cpp request failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    sleep_s,
                )
                if self.reconnect_on_failure:
                    self._reconnect_client()
                if sleep_s > 0:
                    time.sleep(sleep_s)
        raise RuntimeError("unreachable retry state") from last_exc

    def _reconnect_client(self) -> None:
        try:
            self._client.close()
        except Exception:
            logger.debug("Ignoring error while closing llama.cpp client", exc_info=True)
        self._client = self._client_factory(self.timeout)
