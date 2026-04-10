import logging
from typing import Literal

from PIL import Image

from app.inference.scene_graph.reltr_backend import RelTRSceneGraphGenerator
from app.inference.scene_graph.rules_backend import RuleSceneGraphGenerator
from app.inference.scene_graph.vlm_backend import VLMSceneGraphGenerator
from app.inference.types import InferenceDetectionObject
from app.inference.types import SceneGraph

logger = logging.getLogger(__name__)


class SceneGraphService:
    def __init__(
        self,
        mode: Literal["vlm", "rules", "hybrid", "reltr"],
        vlm_backend: VLMSceneGraphGenerator,
        rule_backend: RuleSceneGraphGenerator,
        reltr_backend: RelTRSceneGraphGenerator,
    ):
        self.mode = mode
        self.vlm_backend = vlm_backend
        self.rule_backend = rule_backend
        self.reltr_backend = reltr_backend

    async def generate(
        self,
        detections: list[InferenceDetectionObject],
        *,
        som_image=None,
        raw_image: Image.Image | None = None,
        caption_text: str | None = None,
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
            case "reltr":
                return await self.reltr_backend.generate(raw_image, detections)
            case "vlm":
                if vlm_image is None:
                    return SceneGraph()
                return await self.vlm_backend.generate(
                    vlm_image, detections, caption_text
                )
            case _:

                vlm_graph = (
                    SceneGraph()
                    if vlm_image is None
                    else await self.vlm_backend.generate(
                        vlm_image, detections, caption_text
                    )
                )

                logger.debug("HYBRID SGG, VLM output: %s", vlm_graph)
                rules_graph = self.rule_backend.generate(raw_image, detections)
                logger.debug("HYBRID SGG, RULES output: %s", rules_graph)
                reltr_graph = await self.reltr_backend.generate(raw_image, detections)
                logger.debug("HYBRID SGG, RELTR output: %s", reltr_graph)
                out = vlm_graph + rules_graph + reltr_graph
                logger.debug("HYBRID SGG, VLM + RULES + RELTR output: %s", out)
                return out
