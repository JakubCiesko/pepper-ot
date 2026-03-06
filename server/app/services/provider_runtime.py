import os
from typing import Any

from app.schemas.config import LLMConfig


def resolve_api_key(
    config: LLMConfig,
    *,
    default_env: str,
    default_value: str = "",
) -> str:
    api_key_env = config.api_key_env or default_env
    return os.getenv(api_key_env, default_value)


def build_openai_async_client_kwargs(
    config: LLMConfig,
    *,
    default_api_env: str = "OPENAI_API_KEY",
    default_api_value: str = "EMPTY",
) -> dict[str, Any]:
    client_kwargs = dict(config.client_init_kwargs or {})
    client_kwargs.setdefault(
        "api_key",
        resolve_api_key(
            config,
            default_env=default_api_env,
            default_value=default_api_value,
        ),
    )
    if config.base_url and "base_url" not in client_kwargs:
        client_kwargs["base_url"] = config.base_url
    if config.timeout_seconds is not None and "timeout" not in client_kwargs:
        client_kwargs["timeout"] = config.timeout_seconds
    return client_kwargs


def build_gemini_client_kwargs(
    config: LLMConfig,
    *,
    default_api_env: str = "GEMINI_API_KEY",
    default_api_value: str = "",
) -> dict[str, Any]:
    client_kwargs = dict(config.client_init_kwargs or {})
    client_kwargs.setdefault(
        "api_key",
        resolve_api_key(
            config,
            default_env=default_api_env,
            default_value=default_api_value,
        ),
    )

    if config.timeout_seconds is not None:
        try:
            from google.genai import types  # type: ignore

            client_kwargs.setdefault(
                "http_options", types.HttpOptions(timeout=config.timeout_seconds)
            )
        except Exception:
            pass

    return client_kwargs
