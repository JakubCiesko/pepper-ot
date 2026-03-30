from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types

from app.providers.model_io_common import parse_structured_text
from app.providers.model_io_common import resolve_structured_mode
from app.providers.model_io_common import schema_to_json_schema
from app.providers.vlm.base import BaseVLMClient
from app.schemas.config import LLMConfig

logger = logging.getLogger(__name__)


class GeminiVLMClient(BaseVLMClient):
    def __init__(self, config: LLMConfig, client_kwargs: dict | None = None):
        self.config = config
        self.client = genai.Client(**(client_kwargs or {}))
        logger.info("GeminiVLMClient initialized model=%s", config.model_id)

    def prepare_input(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None,
        output_schema: Any | None = None,
    ) -> dict[str, Any]:

        contents: list[str | types.Part] = [user_prompt]

        if image is None or len(image) == 0:
            logger.warning("Running Gemini VLM without image (None or empty)")
        else:
            try:
                contents.append(
                    types.Part.from_bytes(data=image, mime_type="image/jpeg")
                )
            except Exception as e:
                logger.warning(
                    "Invalid image bytes, falling back to text-only. error=%s", e
                )

        return {"contents": contents}

    async def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None,
        *,
        output_schema: Any | None = None,
        call_overrides: dict[str, Any] | None = None,
    ) -> tuple[str, Any | None]:

        kwargs = self.prepare_kwargs(call_overrides)
        req = self.prepare_input(system_prompt, user_prompt, image, output_schema)

        contents = req["contents"]

        mode = resolve_structured_mode(
            self.config,
            output_schema=output_schema,
            supports_native_structured=True,
            provider_name="gemini_vlm",
        )

        generation_config = kwargs.pop("generate_content_config", {})

        if system_prompt.strip():
            generation_config.setdefault("system_instruction", system_prompt.strip())

        if mode == "provider_native" and output_schema:
            schema_dict = schema_to_json_schema(output_schema)
            if schema_dict:
                generation_config["response_mime_type"] = "application/json"
                generation_config["response_json_schema"] = schema_dict

        elif mode == "parse_output" and output_schema:
            generation_config["response_mime_type"] = "application/json"

        response = await self.client.aio.models.generate_content(
            model=self.config.model_id,
            contents=contents,
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
