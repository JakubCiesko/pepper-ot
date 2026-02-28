from abc import ABC
from abc import abstractmethod
import base64
import io
import logging

from openai import AsyncOpenAI
from PIL import Image
from qwen_vl_utils import process_vision_info
import torch
from transformers import AutoModelForVision2Seq
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from transformers import Qwen3VLForConditionalGeneration

from app.schemas.config import LLMConfig

logger = logging.getLogger(__name__)


class BaseVLMClient(ABC):
    @abstractmethod
    async def infer(self, system_prompt: str, user_prompt: str, image: bytes) -> str:
        pass


class OpenAIVLMClient(BaseVLMClient):
    def __init__(self, config: LLMConfig, client_kwargs: dict | None = None):
        self.config = config
        self.client = AsyncOpenAI(**(client_kwargs or {}))

    async def infer(self, system_prompt: str, user_prompt: str, image: bytes) -> str:
        inference = self.config.inference or {}
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
            temperature=inference.get("temperature", 0.0),
            max_tokens=inference.get("max_tokens", 512),
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""


class LocalHFVLMClient(BaseVLMClient):
    def __init__(
        self,
        config: LLMConfig,
        dtype=None,
        attn_implementation: str = "flash_attention_2",
        trust_remote_code: bool = True,
    ):
        self.config = config
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_id = config.model_id
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
            trust_remote_code=trust_remote_code,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.device = self.model.device

    async def infer(self, system_prompt: str, user_prompt: str, image: bytes) -> str:
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
        ).to(self.device)

        max_new_tokens = int((self.config.inference or {}).get("max_tokens", 512))
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, out, strict=True)]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


class Local4BitVLMClient(BaseVLMClient):
    def __init__(
        self,
        config: LLMConfig,
        trust_remote_code: bool = True,
    ):
        self.config = config
        model_id = config.model_id
        requested_device = config.device
        device_map = {"": requested_device} if requested_device else "auto"

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
        max_new_tokens = int((self.config.inference or {}).get("max_tokens", 512))
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        trimmed = [o[len(i) :] for i, o in zip(inputs["input_ids"], out, strict=True)]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def build_vlm_client(config: LLMConfig) -> BaseVLMClient:
    backend_kwargs = (config.inference or {}).get("backend_kwargs", {})
    logger.info(
        f"Building VLM client backend={config.backend} model={config.model_id} device={config.device}"
    )
    if config.backend == "openai":
        return OpenAIVLMClient(config, client_kwargs=backend_kwargs)
    if config.backend == "local":
        return LocalHFVLMClient(config, **backend_kwargs)
    return Local4BitVLMClient(config, **backend_kwargs)
