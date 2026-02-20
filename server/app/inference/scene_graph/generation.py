import base64
import io
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel
from pydantic import Field

from ..types import SceneGraph
from .vlm import BaseVLM
from .vlm import LLMLabelerConfig
from .vlm import Local4BitVLM
from .vlm import LocalHFVLM
from .vlm import OpenAIVLM
from .vlm import VLMBackend

logger = logging.getLogger(__name__)


class OntologyConfig(BaseModel):
    """Defines world model for teacher and student models in Object Detection FT process.
    Serves as ontology definition for predicates in VLM finetune too (optional param).
    """

    objects: dict[str, str] | None = Field(
        None, description="Map of more specific to more general"
    )
    predicates: list[str] | None = Field(
        None, description="List of predicates in VLM SGG finetuning."
    )

    # TODO: remove this possibly
    # @model_validator(mode="after")
    # def check_at_least_one_ontology(self):
    #     if self.objects or self.predicates:
    #         return self
    #     raise ValueError(
    #         "At least one ontology is required (dict[str, str]). Specify objects or predicates (list[str])."
    #     )


class SceneGraphGenerator:
    def __init__(
        self,
        config: LLMLabelerConfig,
        predicates: list[str] | None = None,
        objects: dict[str, str] | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ):
        self.config = config
        self.predicates = predicates
        self.objects = objects
        self.system_prompt = system_prompt or config.system_prompt
        self.user_prompt = user_prompt
        self.vlm = self.build_vlm(config)

    @staticmethod
    def build_vlm(config: LLMLabelerConfig) -> BaseVLM:
        match config.backend:
            case VLMBackend.OPENAI:
                return OpenAIVLM(config, config.backend_kwargs)
            case VLMBackend.LOCAL:
                return LocalHFVLM(config.model_id, **config.backend_kwargs)
            case VLMBackend.LOCAL_4BIT:
                return Local4BitVLM(config.model_id, **config.backend_kwargs)

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        with image_path.open("rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _to_bytes(image: Path | bytes | Image.Image | np.ndarray) -> bytes:
        """Convert input to bytes for VLM inference."""
        if isinstance(image, bytes):
            return image
        elif isinstance(image, Path):
            return image.read_bytes()
        elif isinstance(image, Image.Image):
            with io.BytesIO() as buf:
                image.save(buf, format="JPEG")
                return buf.getvalue()
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype("uint8"))
            with io.BytesIO() as buf:
                pil_img.save(buf, format="JPEG")
                return buf.getvalue()
        else:
            raise TypeError(
                f"Unsupported input type: {type(image)}. Must be Path, bytes, PIL.Image, or np.ndarray."
            )

    async def generate(
        self, image: Path | bytes | Image.Image | np.ndarray, verbose: bool = False
    ) -> SceneGraph:
        image_bytes = self._to_bytes(image)

        system_prompt = self.system_prompt

        if self.user_prompt:
            user_prompt = self.user_prompt
        elif self.predicates:
            user_prompt = "Allowed predicates: " + ", ".join(self.predicates)
        else:
            user_prompt = "Focus on spatial, semantic, and functional relationships."

        raw = await self.vlm.infer(system_prompt, user_prompt, image_bytes)
        if verbose:
            logger.info(f"VLM output: {raw}")
        try:
            data = json.loads(raw)

            if isinstance(data, dict):
                for key in ["relationships", "scene_graph", "triplets", "relations"]:
                    if key in data:
                        data = data[key]
                        break
            else:
                data = [data]
            if not isinstance(data, list):
                data = []
        except Exception:
            logger.warning("Failed to parse VLM output as JSON")
            data = []
        return SceneGraph.from_list(data, raw=raw)

    # async def batch_generate(self, image_paths: list[Path], batch_size: int = 100):
    #     semaphore = asyncio.Semaphore(batch_size)
    #
    #     async def limited_generate(path: Path):
    #         async with semaphore:
    #             result = await self.generate(path)
    #             return path, result
    #
    #     tasks = [limited_generate(p) for p in image_paths]
    #     logger.info(
    #         f"Starting batch generation for {len(tasks)} images with concurrency {batch_size}..."
    #     )
    #     results = await asyncio.gather(*tasks)
    #     return results
