from __future__ import annotations

import base64
import logging
from typing import Any

from openai import AsyncOpenAI

from app.core.config.llm_contracts import normalize_call_kwargs
from app.core.config.llm_contracts import normalize_openai_parse_kwargs
from app.providers.model_io_common import extract_text_content
from app.providers.model_io_common import extract_text_from_openai_response
from app.providers.model_io_common import parse_structured_text
from app.providers.model_io_common import resolve_structured_mode
from app.providers.model_io_common import validate_parsed_output
from app.providers.vlm.base import BaseVLMClient
from app.schemas.config import LLMConfig

logger = logging.getLogger(__name__)


class OpenAIVLMClient(BaseVLMClient):
    def __init__(
        self,
        config: LLMConfig,
        client_kwargs: dict | None = None,
        supports_native_structured: bool = True,
    ):
        self.config = config
        self.client = AsyncOpenAI(**(client_kwargs or {}))
        self.supports_native_structured = supports_native_structured
        logger.info(
            "OpenAIVLMClient initialized provider=%s model=%s",
            config.provider,
            config.model_id,
        )

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
        kwargs = dict(self.config.call_kwargs or {})
        if call_overrides:
            kwargs.update(call_overrides)
        kwargs = normalize_call_kwargs(self.config.provider, kwargs)
        mode = resolve_structured_mode(
            self.config,
            output_schema=output_schema,
            supports_native_structured=self.supports_native_structured,
            provider_name="openai_vlm",
        )

        encoded = base64.b64encode(image).decode("utf-8")
        native_input = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encoded}",
                    },
                ],
            },
        ]
        if mode == "provider_native" and output_schema is not None:
            parse_kwargs = normalize_openai_parse_kwargs(kwargs)
            try:
                response = await self.client.responses.parse(
                    model=self.config.model_id,
                    input=native_input,
                    text_format=output_schema,
                    **parse_kwargs,
                )
                parsed_native = getattr(response, "output_parsed", None)
                text = extract_text_from_openai_response(response)
                parsed = validate_parsed_output(
                    parsed_native,
                    output_schema,
                    strict=self.config.structured_output.strict,
                )
                return text, parsed
            except Exception as exc:
                logger.warning(
                    "OpenAI VLM provider_native structured call failed, falling back to parse_output provider=%s model=%s error=%s",
                    self.config.provider,
                    self.config.model_id,
                    exc,
                )

        if mode == "parse_output" and output_schema is not None:
            kwargs.setdefault("response_format", {"type": "json_object"})

        response = await self.client.chat.completions.create(
            model=self.config.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                        },
                    ],
                },
            ],
            **kwargs,
        )
        content = response.choices[0].message.content if response.choices else ""
        text = extract_text_content(content)
        parsed = parse_structured_text(
            text,
            output_schema,
            strict=self.config.structured_output.strict,
        )
        return text, parsed
