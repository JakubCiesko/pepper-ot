import io
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from app.inference.types import SceneGraph
from app.schemas.config import SceneGraphVLMConfig
from app.schemas.scene import SceneGraphStructuredResponse
from app.services.vlm_client import BaseVLMClient
from app.services.vlm_client import build_vlm_client

logger = logging.getLogger(__name__)


class VLMSceneGraphBackend:
    def __init__(
        self,
        config: SceneGraphVLMConfig,
        predicates: list[str] | None = None,
        objects: dict[str, str] | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ):
        self.config = config
        self.predicates = predicates
        self.objects = objects
        self.system_prompt = system_prompt or ""
        self.user_prompt = user_prompt
        self.client: BaseVLMClient = build_vlm_client(config)

    def update_runtime(
        self,
        config: SceneGraphVLMConfig,
        predicates: list[str] | None,
        objects: dict[str, str] | None,
        system_prompt: str | None,
        user_prompt: str | None,
        rebuild_client: bool = False,
    ):
        self.config = config
        self.predicates = predicates
        self.objects = objects
        self.system_prompt = system_prompt or ""
        self.user_prompt = user_prompt
        if rebuild_client:
            self.client = build_vlm_client(config)
        else:
            self.client.update_runtime(config)

    @staticmethod
    def _to_bytes(image: Path | bytes | Image.Image | np.ndarray) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, Path):
            return image.read_bytes()
        if isinstance(image, Image.Image):
            with io.BytesIO() as buf:
                image.save(buf, format="JPEG")
                return buf.getvalue()
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype("uint8"))
            with io.BytesIO() as buf:
                pil_img.save(buf, format="JPEG")
                return buf.getvalue()
        raise TypeError(
            f"Unsupported input type: {type(image)}. Must be Path, bytes, PIL.Image, or np.ndarray."
        )

    @staticmethod
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

    @staticmethod
    def _normalize_data(data):
        if isinstance(data, dict):
            for key in ["relationships", "scene_graph", "triplets", "relations"]:
                if key in data:
                    data = data[key]
                    break
            else:
                if all(k in data for k in ["sub", "rel", "obj"]):
                    data = [data]
        elif not isinstance(data, list):
            data = [data]
        return data if isinstance(data, list) else []

    def _parse_json(self, raw: str) -> list[dict]:
        try:
            data = json.loads(raw)
            return self._normalize_data(data)
        except Exception:
            extracted = self._extract_json_block(raw)
            if extracted:
                try:
                    data = json.loads(extracted)
                    return self._normalize_data(data)
                except Exception:
                    return []
            return []

    async def _repair(self, image_bytes: bytes, raw: str) -> str:
        repair_system = (
            "You are a JSON repair engine. Return ONLY valid JSON with key "
            '"relationships" that is a list of {"sub","rel","obj"} objects.'
        )
        clipped = raw[:2000]
        repair_user = (
            "Fix the following output into valid JSON only. No extra text.\n\n"
            f"OUTPUT:\n{clipped}"
        )
        repaired, _ = await self.client.infer(repair_system, repair_user, image_bytes)
        return repaired

    def _build_user_prompt(self) -> str:
        if self.user_prompt:
            if self.predicates and "{predicates}" in self.user_prompt:
                return self.user_prompt.replace(
                    "{predicates}", ", ".join(self.predicates)
                )
            return self.user_prompt
        if self.predicates:
            return "Allowed predicates: " + ", ".join(self.predicates)
        return "Focus on spatial, semantic, and functional relationships."

    async def generate(
        self, image: Path | bytes | Image.Image | np.ndarray
    ) -> SceneGraph:
        image_bytes = self._to_bytes(image)
        user_prompt = self._build_user_prompt()
        try:
            raw, parsed = await self.client.infer(
                self.system_prompt,
                user_prompt,
                image_bytes,
                output_schema=SceneGraphStructuredResponse,
            )
        except Exception as exc:
            logger.warning(
                f"Structured VLM generation failed, falling back to raw mode: {exc}"
            )
            raw, parsed = await self.client.infer(
                self.system_prompt,
                user_prompt,
                image_bytes,
                output_schema=None,
            )
        # this is just for precommit to shutup
        data: list[dict] = []
        if isinstance(parsed, SceneGraphStructuredResponse):
            data = [rel.model_dump() for rel in parsed.relationships]
        elif parsed is not None:
            data = self._normalize_data(parsed)
        else:
            data = self._parse_json(raw)
        if not data:
            logger.warning("Failed to parse VLM output as JSON, attempting repair")
            repaired = await self._repair(image_bytes, raw)
            data = self._parse_json(repaired)
            if not data:
                logger.warning("VLM repair failed, returning empty scene graph")
        logger.debug(f"User VLM Prompt: {user_prompt}")
        logger.debug(f"Raw VLM Response: {raw}")
        return SceneGraph.from_list(data, raw=raw)
