from abc import ABC
from abc import abstractmethod
import base64
import io
import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI
from PIL import Image
from pydantic import TypeAdapter
import torch
from transformers import AutoModelForVision2Seq
from transformers import AutoProcessor

from app.schemas.config import LLMConfig

logger = logging.getLogger(__name__)


class BaseVLMClient(ABC):
    @abstractmethod
    async def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes,
        *,
        output_schema: Any | None = None,
        call_overrides: dict[str, Any] | None = None,
    ) -> tuple[str, Any | None]:
        pass


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
        mode = _resolve_structured_mode(
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
            parse_kwargs = dict(kwargs)
            if "max_tokens" in parse_kwargs and "max_output_tokens" not in parse_kwargs:
                parse_kwargs["max_output_tokens"] = parse_kwargs.pop("max_tokens")
            try:
                response = await self.client.responses.parse(
                    model=self.config.model_id,
                    input=native_input,
                    text_format=output_schema,
                    **parse_kwargs,
                )
                parsed_native = getattr(response, "output_parsed", None)
                text = _extract_text_from_openai_response(response)
                parsed = _validate_parsed_output(
                    parsed_native,
                    output_schema,
                    strict=self.config.structured_output.strict,
                )
                return text, parsed
            except Exception as exc:
                logger.warning(
                    "OpenAI VLM provider_native structured call failed, falling back to parse_output: %s",
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
        text = _extract_text(content)
        parsed = _parse_structured_text(
            text,
            output_schema,
            strict=self.config.structured_output.strict,
        )
        return text, parsed


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
        mode = _resolve_structured_mode(
            self.config,
            output_schema=output_schema,
            supports_native_structured=True,
            provider_name="gemini_vlm",
        )

        generation_config = kwargs.pop("generate_content_config", {})
        generation_config.setdefault("system_instruction", system_prompt)
        if mode == "provider_native" and output_schema is not None:
            schema_dict = _schema_to_json_schema(output_schema)
            if schema_dict is not None:
                generation_config.setdefault("response_mime_type", "application/json")
                generation_config.setdefault("response_json_schema", schema_dict)
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
        parsed = _parse_structured_text(
            text,
            output_schema,
            strict=self.config.structured_output.strict,
        )
        return text, parsed


class LocalHFVLMClient(BaseVLMClient):
    """
    Generic local HF VLM runner. It intentionally avoids model-family guessing and
    relies on Auto* classes with optional overrides in config kwargs.
    """

    def __init__(
        self,
        config: LLMConfig,
        trust_remote_code: bool = True,
    ):
        self.config = config
        requested_device = config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dtype = torch.bfloat16 if requested_device.startswith("cuda") else torch.float32

        model_kwargs = dict(config.client_init_kwargs or {})
        model_kwargs.setdefault("trust_remote_code", trust_remote_code)
        model_kwargs.setdefault("torch_dtype", dtype)

        if "device_map" not in model_kwargs:
            model_kwargs["device_map"] = (
                "auto" if requested_device.startswith("cuda") else {"": "cpu"}
            )

        self.model = AutoModelForVision2Seq.from_pretrained(
            config.model_id,
            **model_kwargs,
        )

        processor_kwargs = dict(config.client_init_kwargs.get("processor_kwargs", {}))
        processor_kwargs.setdefault("trust_remote_code", trust_remote_code)
        self.processor = AutoProcessor.from_pretrained(
            config.model_id,
            **processor_kwargs,
        )

        self.device = (
            self.model.device
            if hasattr(self.model, "device")
            else torch.device(requested_device)
        )

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

        max_new_tokens = int(
            kwargs.pop("max_new_tokens", kwargs.pop("max_tokens", 512))
        )

        img = Image.open(io.BytesIO(image)).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        try:
            text_input = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.processor(
                text=[text_input],
                images=[img],
                return_tensors="pt",
            )
        except Exception:
            fallback_prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"
            model_inputs = self.processor(
                text=[fallback_prompt],
                images=[img],
                return_tensors="pt",
            )

        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }

        with torch.no_grad():
            out = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)

        input_ids = model_inputs.get("input_ids")
        if input_ids is not None:
            trimmed = [o[len(i) :] for i, o in zip(input_ids, out, strict=True)]
        else:
            trimmed = out

        text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        parsed = _parse_structured_text(
            text,
            output_schema,
            strict=self.config.structured_output.strict,
        )
        return text, parsed


class Local4BitVLMClient(LocalHFVLMClient):
    pass


def build_vlm_client(config: LLMConfig) -> BaseVLMClient:
    logger.info(
        "Building VLM client provider=%s model=%s device=%s",
        config.provider,
        config.model_id,
        config.device,
    )

    provider = config.provider

    if provider in {"openai", "openai_compatible"}:
        client_kwargs = dict(config.client_init_kwargs or {})
        api_key_env = config.api_key_env or "OPENAI_API_KEY"
        client_kwargs.setdefault("api_key", os.getenv(api_key_env, "EMPTY"))
        if config.base_url and "base_url" not in client_kwargs:
            client_kwargs["base_url"] = config.base_url
        if config.timeout_seconds is not None and "timeout" not in client_kwargs:
            client_kwargs["timeout"] = config.timeout_seconds
        return OpenAIVLMClient(
            config,
            client_kwargs=client_kwargs,
            supports_native_structured=(provider == "openai"),
        )

    if provider == "gemini":
        client_kwargs = dict(config.client_init_kwargs or {})
        api_key_env = config.api_key_env or "GEMINI_API_KEY"
        client_kwargs.setdefault("api_key", os.getenv(api_key_env, ""))
        return GeminiVLMClient(config, client_kwargs=client_kwargs)

    if provider in {"local_hf", "local_4bit"}:
        if provider == "local_4bit":
            return Local4BitVLMClient(config)
        return LocalHFVLMClient(config)

    raise ValueError(f"Unsupported VLM provider: {provider}")


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(filter(None, parts))
    return str(content or "")


def _extract_json_block(raw: str) -> str | None:
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return None


def _parse_structured_text(raw: str, schema: Any | None, strict: bool) -> Any | None:
    if schema is None:
        return None
    block = _extract_json_block(raw)
    if block is None:
        if strict:
            raise ValueError("Structured output requested but no JSON found")
        return None
    parsed = json.loads(block)
    adapter = TypeAdapter(schema)
    return adapter.validate_python(parsed)


def _validate_parsed_output(
    parsed: Any, schema: Any | None, strict: bool
) -> Any | None:
    if schema is None:
        return None
    if parsed is None:
        if strict:
            raise ValueError("Structured output is empty")
        return None
    adapter = TypeAdapter(schema)
    return adapter.validate_python(parsed)


def _schema_to_json_schema(schema: Any | None) -> dict[str, Any] | None:
    if schema is None:
        return None
    try:
        return TypeAdapter(schema).json_schema()
    except Exception:
        return None


def _resolve_structured_mode(
    config: LLMConfig,
    *,
    output_schema: Any | None,
    supports_native_structured: bool,
    provider_name: str,
) -> str:
    if output_schema is None:
        return "parse_output"
    mode = config.structured_output.mode
    if mode == "provider_native" and not supports_native_structured:
        logger.warning(
            "Structured mode provider_native requested for provider=%s but not supported; using parse_output",
            provider_name,
        )
        return "parse_output"
    return mode


def _extract_text_from_openai_response(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    output = getattr(response, "output", None)
    if not output:
        return ""
    parts: list[str] = []
    for item in output:
        content = getattr(item, "content", None) or []
        for chunk in content:
            chunk_text = getattr(chunk, "text", None)
            if chunk_text:
                parts.append(str(chunk_text))
    return "\n".join(parts)
