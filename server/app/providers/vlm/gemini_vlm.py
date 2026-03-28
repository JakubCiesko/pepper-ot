from __future__ import annotations

import logging
from typing import Any

from app.core.config.llm_contracts import normalize_call_kwargs
from app.providers.model_io_common import parse_structured_text
from app.providers.model_io_common import resolve_structured_mode
from app.providers.model_io_common import schema_to_json_schema
from app.providers.vlm.base import BaseVLMClient
from app.schemas.config import LLMConfig

logger = logging.getLogger(__name__)


class GeminiVLMClient(BaseVLMClient):
    def __init__(self, config: LLMConfig, client_kwargs: dict | None = None):
        self.config = config
        try:
            from google import genai  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Gemini provider requested but google-genai is unavailable"
            ) from exc
        self.client = genai.Client(**(client_kwargs or {}))
        logger.info("GeminiVLMClient initialized model=%s", config.model_id)

    def update_runtime(self, config: LLMConfig):
        self.config = config

    async def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes,
        *,
        output_schema: Any | None = None,
        call_overrides: dict[str, Any] | None = None,
    ) -> tuple[str, Any | None]:
        try:
            from google.genai import types  # type: ignore
        except Exception as exc:
            raise RuntimeError("Gemini types are unavailable") from exc

        kwargs = dict(self.config.call_kwargs or {})
        if call_overrides:
            kwargs.update(call_overrides)
        kwargs = normalize_call_kwargs(self.config.provider, kwargs)
        mode = resolve_structured_mode(
            self.config,
            output_schema=output_schema,
            supports_native_structured=True,
            provider_name="gemini_vlm",
        )

        generation_config = kwargs.pop("generate_content_config", {})
        if system_prompt and system_prompt.strip():
            generation_config.setdefault("system_instruction", system_prompt.strip())
        if mode == "provider_native" and output_schema is not None:
            schema_dict = schema_to_json_schema(output_schema)
            if schema_dict is not None:
                generation_config.setdefault("response_mime_type", "application/json")
                generation_config.setdefault("response_json_schema", schema_dict)
            else:
                logger.warning(
                    "Gemini VLM provider_native requested but schema conversion failed; continuing with parse_output-compatible parsing"
                )
        elif mode == "parse_output" and output_schema is not None:
            generation_config.setdefault("response_mime_type", "application/json")

        response = await self.client.aio.models.generate_content(
            model=self.config.model_id,
            contents=[
                types.Part.from_bytes(data=image, mime_type="image/jpeg"),
                user_prompt,
            ],
            config=types.GenerateContentConfig(**generation_config),
            **kwargs,
        )
        text = str(getattr(response, "text", "") or "")
        parsed = parse_structured_text(
            text,
            output_schema,
            strict=self.config.structured_output.strict,
        )
        return text, parsed
