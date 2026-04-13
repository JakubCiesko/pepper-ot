import logging

from PIL import Image

from app.inference.scene_graph.reltr_backend import RelTRSceneGraphGenerator
from app.inference.scene_graph.rules_backend import RuleSceneGraphGenerator
from app.inference.scene_graph.vlm_backend import VLMSceneGraphGenerator
from app.inference.types import InferenceDetectionObject
from app.inference.types import SceneGraph
from app.schemas.scene import SceneState

logger = logging.getLogger(__name__)


class SceneGraphService:
    def __init__(
        self,
        vlm_backend: VLMSceneGraphGenerator,
        rule_backend: RuleSceneGraphGenerator,
        reltr_backend: RelTRSceneGraphGenerator,
    ):
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
        scene_state: SceneState | None = None,
    ) -> SceneGraph:
        vlm_image = som_image if som_image is not None else raw_image
        enabled_backends = self._enabled_backends()
        logger.info(
            "Generating scene graph with backends=%s merge_strategy='union+dedup' for %d detections",
            ",".join(enabled_backends) if enabled_backends else "none",
            len(detections),
        )
        graph = SceneGraph()

        if self.rule_backend.rules_config.enabled:
            rules_graph = self.rule_backend.generate(raw_image, detections)
            logger.debug("SGG RULES output: %s", rules_graph)
            graph = graph + rules_graph

        if self.reltr_backend.config.enabled:
            reltr_graph = await self.reltr_backend.generate(raw_image, detections)
            logger.debug("SGG RELTR output: %s", reltr_graph)
            graph = graph + reltr_graph

        if self.vlm_backend.config.enabled:
            vlm_graph = (
                SceneGraph()
                if vlm_image is None
                else await self.vlm_backend.generate(
                    vlm_image, detections, caption_text
                )
            )
            logger.debug("SGG VLM output: %s", vlm_graph)
            graph = graph + vlm_graph

        logger.debug("Merged SGG output: %s", graph)
        enhanced_graph = self.enhance_scene_graph_with_robot_data(
            graph, detections, scene_state
        )
        logger.debug("Final Robot-Enhanced Graph: %s", enhanced_graph)
        return enhanced_graph

    def _enabled_backends(self) -> list[str]:
        names: list[str] = []
        if self.rule_backend.rules_config.enabled:
            names.append("rules")
        if self.reltr_backend.config.enabled:
            names.append("reltr")
        if self.vlm_backend.config.enabled:
            names.append("vlm")
        return names

    def enhance_scene_graph_with_robot_data(
        self,
        graph: SceneGraph,
        detections: list[InferenceDetectionObject],
        scene_state: SceneState,
    ) -> SceneGraph:
        if not scene_state or not detections:
            return graph
        current_ids = {
            int(det.object_id) for det in detections if isinstance(det.object_id, int)
        }
        current_objects = [obj for obj in scene_state.objects if obj.id in current_ids]
        label_edges = [
            {
                "sub": f"{obj.label}_{obj.id}",
                "rel": attribute,
                "obj": f"{obj.label}_{obj.id}",
            }
            for obj in current_objects
            for attribute in obj.attributes
        ]
        robot_metadata_scene_graph = SceneGraph.from_list(label_edges)
        return graph + robot_metadata_scene_graph
