from __future__ import annotations

import logging

from app.providers.common.runtime_setup import build_gemini_client_kwargs
from app.providers.common.runtime_setup import build_openai_async_client_kwargs
from app.providers.vlm.base import BaseVLMClient
from app.providers.vlm.gemini_vlm import GeminiVLMClient
from app.providers.vlm.local_hf_vlm import Local4BitVLMClient
from app.providers.vlm.local_hf_vlm import LocalHFVLMClient
from app.providers.vlm.openai_vlm import OpenAIVLMClient
from app.schemas.config import LLMConfig

logger = logging.getLogger(__name__)


def build_vlm_client(config: LLMConfig) -> BaseVLMClient:
    logger.info(
        "Building VLM client provider=%s model=%s device=%s",
        config.provider,
        config.model_id,
        config.device,
    )

    provider = config.provider

    if provider in {"openai", "openai_compatible"}:
        client_kwargs = build_openai_async_client_kwargs(
            config,
            default_api_env="OPENAI_API_KEY",
            default_api_value="EMPTY",
        )
        return OpenAIVLMClient(
            config,
            client_kwargs=client_kwargs,
            supports_native_structured=(provider == "openai"),
        )

    if provider == "gemini":
        client_kwargs = build_gemini_client_kwargs(
            config,
            default_api_env="GEMINI_API_KEY",
            default_api_value="",
        )
        return GeminiVLMClient(config, client_kwargs=client_kwargs)

    if provider in {"local_hf", "local_4bit"}:
        if provider == "local_4bit":
            return Local4BitVLMClient(config)
        return LocalHFVLMClient(config)

    raise ValueError(f"Unsupported VLM provider: {provider}")
