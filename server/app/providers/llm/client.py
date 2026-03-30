import logging
from typing import Any

from google import genai
from openai import AsyncOpenAI

from app.providers.common.runtime_setup import build_gemini_client_kwargs
from app.providers.common.runtime_setup import build_openai_async_client_kwargs
from app.providers.llm.base import BaseTextProvider
from app.providers.llm.base import LLMResponse
from app.providers.llm.gemini_llm import GeminiTextProvider
from app.providers.llm.hf_llm import LocalHFTextProvider
from app.providers.llm.openai_llm import OpenAITextProvider
from app.schemas.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Provider-agnostic text generation client.

    Supports OpenAI, Gemini, and OpenAI-compatible endpoints (vLLM/Ollama-like APIs).
    Runtime config updates are supported via update_runtime().
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider: BaseTextProvider | None = None
        self.update_runtime(config, rebuild_client=True)

    def update_runtime(self, config: LLMConfig, rebuild_client: bool = True):
        self.config = config
        if rebuild_client or self.provider is None:
            self.provider = self._build_provider(config)

    def _build_provider(self, config: LLMConfig) -> BaseTextProvider:
        provider = config.provider

        if provider in {"openai", "openai_compatible"}:
            client_kwargs = build_openai_async_client_kwargs(
                config,
                default_api_env="OPENAI_API_KEY",
                default_api_value="EMPTY",
            )
            return OpenAITextProvider(
                AsyncOpenAI(**client_kwargs),
                supports_native_structured=(provider == "openai"),
            )

        if provider in {"local_hf", "local_4bit"}:
            return LocalHFTextProvider(config)

        if provider == "gemini":
            client_kwargs = build_gemini_client_kwargs(
                config,
                default_api_env="GEMINI_API_KEY",
                default_api_value="",
            )
            return GeminiTextProvider(genai.Client(**client_kwargs))

        raise ValueError(f"Unsupported LLM provider: {provider}")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        output_schema: Any | None = None,
        call_overrides: dict[str, Any] | None = None,
    ) -> LLMResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.provider is None:
            self.update_runtime(self.config, rebuild_client=True)
        return await self.provider.generate(
            config=self.config,
            messages=messages,
            output_schema=output_schema,
            call_overrides=call_overrides,
        )

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = await self.generate(system_prompt, user_prompt)
            response = response.text or "I'm not sure what to say."
            logger.info(
                "LLMClient Text Generation, SYSTEM_PROMPT=[%s], USER_PROMPT=[%s], LLM_OUTPUT=[%s]",
                system_prompt,
                user_prompt,
                response,
            )
            return response
        except Exception as exc:
            logger.error(f"LLM generation error: {exc}")
            return "I am having trouble connecting to my language center right now."
