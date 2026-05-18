from __future__ import annotations

import os

from src.llm_backend.base import LLMBackend


def create_backend(
    backend: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    **kwargs,
) -> LLMBackend:
    """Factory that creates the appropriate LLM backend from config or env vars.

    Priority: explicit args > env vars > defaults.
    """
    backend = backend or os.getenv("LLM_BACKEND", "api")
    base_url = base_url or os.getenv("LLM_BASE_URL")
    model = model or os.getenv("LLM_MODEL")
    api_key = api_key or os.getenv("LLM_API_KEY")

    backend = backend.lower()

    if backend == "llamacpp":
        from src.llm_backend.llamacpp_backend import LlamaCppBackend

        return LlamaCppBackend(
            base_url=base_url or "http://localhost:8080",
            model=model or "local-model",
            api_key=api_key or "no-key",
            **kwargs,
        )

    if backend == "vllm":
        from src.llm_backend.vllm_backend import VLLMBackend

        return VLLMBackend(
            base_url=base_url or "http://localhost:8000/v1",
            model=model or "meta-llama/Llama-3.1-70B-Instruct",
            api_key=api_key or "EMPTY",
            **kwargs,
        )

    if backend == "api":
        from src.llm_backend.api_backend import APIBackend

        return APIBackend(
            base_url=base_url,
            model=model or "gpt-4o",
            api_key=api_key,
            **kwargs,
        )

    raise ValueError(f"Unknown LLM backend: {backend!r}. Choose from: llamacpp, vllm, api")
