from abc import ABC
from abc import abstractmethod
import base64
from enum import StrEnum
import io
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel
from pydantic import Field
from qwen_vl_utils import process_vision_info
import torch
from transformers import AutoModelForVision2Seq
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from transformers import Qwen3VLForConditionalGeneration


class VLMBackend(StrEnum):
    OPENAI = "openai"
    LOCAL = "local"
    LOCAL_4BIT = "local_4bit"


# TODO: move elsewhere
class LLMLabelerConfig(BaseModel):
    """Configuration for the VLM Scene Graph ground truth Generator (ex. GPT-4o)."""

    backend: VLMBackend = VLMBackend.OPENAI
    # TODO: Right now, vendor-locked for OpenAI, might change for future
    model_id: str = "gpt-4o"
    path_to_model: Path | None = None
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(512, gt=0)

    system_prompt: str = (
        "You are a robotic scene graph generator. "
        "Analyze the provided Set-of-Mark (SoM) image where objects are marked with numerical IDs. "
        "Output a JSON list of spatial or other relationships "
        # "using ONLY the allowed predicates. " # Provide predicates into prompt if ClosedVocab, else dont mention
        "Unary predicates are represented by the same subject and object "
        "Format: [{'sub': 'ID', 'rel': 'PREDICATE', 'obj': 'ID'}]. "
        "Example: [{'sub': '1', 'rel': 'holding', 'obj': '2'}, {'sub': '1', 'rel': 'red', 'obj': '1'}]."
    )
    backend_kwargs: dict[str, Any] = Field(default_factory=dict)


class BaseVLM(ABC):
    @abstractmethod
    async def infer(self, system_prompt: str, user_prompt: str, image: bytes) -> str:
        pass


class OpenAIVLM(BaseVLM):
    def __init__(self, config: LLMLabelerConfig, openai_client_config: dict = None):
        if openai_client_config is None:
            openai_client_config = {}
        self.client = AsyncOpenAI(**openai_client_config)
        self.config = config

    async def infer(self, system_prompt: str, user_prompt: str, image: bytes) -> str:
        encoded = base64.b64encode(image).decode()
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
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""


# TODO: Rewrite this, too verbose and weird


class LocalHFVLM(BaseVLM):
    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        dtype=None,
        attn_implementation: str = "flash_attention_2",
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if "qwen2" in model_id.lower().strip():
            model_cls = Qwen2VLForConditionalGeneration
        elif "qwen3" in model_id.lower().strip():
            model_cls = Qwen3VLForConditionalGeneration
        else:
            model_cls = AutoModelForVision2Seq
        self.model = model_cls.from_pretrained(
            model_id,
            dtype=dtype or (torch.bfloat16 if device == "cuda" else torch.float32),
            device_map="auto",
            attn_implementation=attn_implementation,
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = self.model.device

    async def infer(self, system_prompt: str, user_prompt: str, image: bytes) -> str:
        img = Image.open(io.BytesIO(image)).convert("RGB")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=512)

        trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, out, strict=True)]

        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


class Local4BitVLM(BaseVLM):
    def __init__(
        self, model_id: str, trust_remote_code: bool = True, device_map: str = "auto"
    ):
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )

    async def infer(self, system_prompt: str, user_prompt: str, image: bytes) -> str:
        img = Image.open(io.BytesIO(image)).convert("RGB")

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=512)

        trimmed = [o[len(i) :] for i, o in zip(inputs["input_ids"], out, strict=True)]

        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
