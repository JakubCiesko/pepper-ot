import logging
from typing import Literal

from app.inference.scene_graph.rules_backend import RuleBasedSceneGraphBackend
from app.inference.scene_graph.vlm_backend import VLMSceneGraphBackend
from app.inference.types import DetectionObject
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
        self, som_image, detections: list[DetectionObject]
    ) -> SceneGraph:
        match self.mode:
            case "rules":
                return self.rule_backend.generate(detections)
            case "vlm":
                return await self.vlm_backend.generate(som_image)
            case _:
                vlm_graph = await self.vlm_backend.generate(som_image)
                rules_graph = self.rule_backend.generate(detections)
                return vlm_graph + rules_graph
