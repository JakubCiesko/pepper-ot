import logging
from typing import Literal

from PIL import Image

from app.inference.scene_graph.rules_backend import RuleBasedSceneGraphBackend
from app.inference.scene_graph.vlm_backend import VLMSceneGraphBackend
from app.inference.types import InferenceDetectionObject
from app.inference.types import SceneGraph

logger = logging.getLogger(__name__)


class SceneGraphService:
    def __init__(
        self,
        mode: Literal["vlm", "rules", "hybrid"],
        vlm_backend: VLMSceneGraphBackend,
        rule_backend: RuleBasedSceneGraphBackend,
    ):
        self.mode = mode
        self.vlm_backend = vlm_backend
        self.rule_backend = rule_backend

    async def generate(
        self,
        detections: list[InferenceDetectionObject],
        *,
        som_image=None,
        raw_image: Image.Image | None = None,
    ) -> SceneGraph:
        vlm_image = som_image if som_image is not None else raw_image
        logger.info(
            "Generating scene graph with mode=%s for %d detections",
            self.mode,
            len(detections),
        )
        # pass raw to rules because colors get distorted by bboxes colors
        match self.mode:
            case "rules":
                return self.rule_backend.generate(raw_image, detections)
            case "vlm":
                if vlm_image is None:
                    return SceneGraph()
                return await self.vlm_backend.generate(vlm_image, detections)
            case _:
                if vlm_image is None:
                    vlm_graph = SceneGraph()
                else:
                    vlm_graph = await self.vlm_backend.generate(vlm_image, detections)
                logger.info("HYBRID SGG, VLM output: %s", vlm_graph)
                rules_graph = self.rule_backend.generate(raw_image, detections)
                logger.info("HYBRID SGG, RULES output: %s", rules_graph)
                out = vlm_graph + rules_graph
                logger.info("HYBRID SGG, VLM + RULES output: %s", out)
                return out
