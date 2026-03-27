from __future__ import annotations

import io
import logging
from typing import Any

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor

from app.core.config.llm_contracts import normalize_call_kwargs
from app.providers.model_io_common import parse_structured_text
from app.providers.vlm.base import BaseVLMClient
from app.schemas.config import LLMConfig

logger = logging.getLogger(__name__)


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
        model_kwargs.setdefault("dtype", dtype)

        if "device_map" not in model_kwargs:
            model_kwargs["device_map"] = (
                "auto" if requested_device.startswith("cuda") else {"": "cpu"}
            )

        self.model = AutoModelForImageTextToText.from_pretrained(
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
        logger.info("LocalHFVLMClient initialized model=%s", config.model_id)

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
