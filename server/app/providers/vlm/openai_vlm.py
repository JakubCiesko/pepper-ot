from __future__ import annotations

import base64
import logging
from typing import Any

import instructor
from openai import AsyncOpenAI

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
        self.instructor_client = instructor.from_openai(
            self.client, mode=instructor.Mode.TOOLS
        )
        self.supports_native_structured = supports_native_structured
        logger.info(
            "OpenAIVLMClient initialized provider=%s model=%s",
            config.provider,
            config.model_id,
        )

    def prepare_input(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None,
        output_schema: Any | None = None,
    ) -> dict[str, Any]:

        encoded = None

        if image is None or len(image) == 0:
            logger.warning("Running OpenAI VLM without image (None or empty)")
        else:
            try:
                encoded = base64.b64encode(image).decode("utf-8")
            except Exception as e:
                logger.warning(
                    "Invalid image bytes, falling back to text-only. error=%s", e
                )

        user_content_native = [{"type": "input_text", "text": user_prompt}]
        user_content_standard = [{"type": "text", "text": user_prompt}]

        if encoded is not None:
            user_content_native.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encoded}",
                }
            )
            user_content_standard.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )

        return {
            "native_input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content_native},
            ],
            "standard_messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content_standard},
            ],
        }

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

        native_input = req["native_input"]
        standard_messages = req["standard_messages"]

        mode = resolve_structured_mode(
            self.config,
            output_schema=output_schema,
            supports_native_structured=self.supports_native_structured,
            provider_name="openai_vlm",
        )

        # ---- Native structured ----
        if mode == "provider_native" and output_schema is not None:
            try:
                response = await self.client.responses.parse(
                    model=self.config.model_id,
                    input=native_input,
                    text_format=output_schema,
                    **normalize_openai_parse_kwargs(kwargs),
                )

                text = extract_text_from_openai_response(response)
                parsed = validate_parsed_output(
                    getattr(response, "output_parsed", None),
                    output_schema,
                    strict=self.config.structured_output.strict,
                )
                return text, parsed

            except Exception as exc:
                logger.warning("Native structured failed → fallback: %s", exc)
                mode = "instructor"

        # ---- Instructor ----
        if mode == "instructor" and output_schema is not None:
            try:
                parsed = await self.instructor_client.chat.completions.create(
                    model=self.config.model_id,
                    messages=standard_messages,
                    response_model=output_schema,
                    **kwargs,
                )
                text = (
                    parsed.model_dump_json()
                    if hasattr(parsed, "model_dump_json")
                    else str(parsed)
                )
                return text, parsed

            except Exception as exc:
                logger.warning("Instructor failed → fallback: %s", exc)
                mode = "parse_output"

        # ---- Plain / JSON ----
        if mode == "parse_output" and output_schema is not None:
            kwargs.setdefault("response_format", {"type": "json_object"})

        response = await self.client.chat.completions.create(
            model=self.config.model_id,
            messages=standard_messages,
            **kwargs,
        )

        text = extract_text_content(
            response.choices[0].message.content if response.choices else ""
        )

        parsed = parse_structured_text(
            text,
            output_schema,
            strict=self.config.structured_output.strict,
        )

        return text, parsed
