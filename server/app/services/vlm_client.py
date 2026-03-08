from abc import ABC
from abc import abstractmethod
import base64
import io
import logging
from typing import Any

from openai import AsyncOpenAI
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq
from transformers import AutoProcessor

from app.core.llm_contracts import normalize_call_kwargs
from app.core.llm_contracts import normalize_openai_parse_kwargs
from app.schemas.config import LLMConfig
from app.services.model_io_common import extract_text_content
from app.services.model_io_common import extract_text_from_openai_response
from app.services.model_io_common import parse_structured_text
from app.services.model_io_common import resolve_structured_mode
from app.services.model_io_common import schema_to_json_schema
from app.services.model_io_common import validate_parsed_output
from app.services.provider_runtime import build_gemini_client_kwargs
from app.services.provider_runtime import build_openai_async_client_kwargs

logger = logging.getLogger(__name__)


class BaseVLMClient(ABC):
    def update_runtime(self, config: LLMConfig):
        self.config = config

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
                    "OpenAI VLM provider_native structured call failed, deterministically falling back to parse_output provider=%s model=%s error=%s",
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
        generation_config.setdefault("system_instruction", system_prompt)
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

    def update_runtime(self, config: LLMConfig):
        # Keep loaded model/processor; hot updates refresh runtime call behavior.
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

        max_new_tokens = int(
            kwargs.pop("max_new_tokens", kwargs.pop("max_tokens", 512))
        )

        img = Image.open(io.BytesIO(image)).convert("RGB")
        hints = getattr(self.config, "local_vlm_hints", None)
        prompt_style = (
            getattr(hints, "prompt_template_style", "auto") if hints else "auto"
        )
        image_token_strategy = (
            getattr(hints, "image_token_strategy", "auto") if hints else "auto"
        )
        hinted_user_prompt = user_prompt
        if image_token_strategy in {"single", "multi"}:
            hinted_user_prompt = (
                f"[IMAGE_TOKEN_STRATEGY={image_token_strategy}]\n{hinted_user_prompt}"
            )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": hinted_user_prompt},
                ],
            },
        ]

        try:
            if prompt_style == "plain":
                raise ValueError("plain prompt style requested")
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
        except Exception as exc:
            fallback_prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"
            logger.info(
                "Local VLM apply_chat_template failed or disabled (style=%s). Using plain fallback prompt. reason=%s",
                prompt_style,
                exc,
            )
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
        parsed = parse_structured_text(
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
